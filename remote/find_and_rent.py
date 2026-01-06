#!/usr/bin/env python
"""
Find and rent the cheapest GPU instance on vast.ai.

Usage:
    python remote/find_and_rent.py                    # Rent cheapest H100_SXM (default max $5/hr)
    python remote/find_and_rent.py --gpu L40S         # Rent cheapest L40S
    python remote/find_and_rent.py --gpu A100_SXM     # Rent cheapest A100 SXM
    python remote/find_and_rent.py --max-price 2.0    # Only consider offers under $2/hr
    python remote/find_and_rent.py --dry-run          # Show what would be rented without renting
"""

import argparse
import subprocess
import json
import sys
import time


def run_cmd(cmd: list[str], capture=True) -> tuple[int, str]:
    """Run a command and return (returncode, output)."""
    result = subprocess.run(cmd, capture_output=capture, text=True)
    output = result.stdout + result.stderr if capture else ""
    return result.returncode, output.strip()


def search_offers(gpu_name: str, num_gpus: int = 1, min_reliability: float = 95.0, max_price: float = None, exclude_countries: list = None, debug: bool = False) -> list[dict]:
    """Search for available GPU offers."""
    # Build query - just gpu_name, filter rest in Python
    query = f"gpu_name={gpu_name}"
    cmd = ["vastai", "search", "offers", query, "--order", "dph", "--raw"]
    exclude_countries = exclude_countries or []

    if debug:
        print(f"DEBUG: Query = {query}")
        print(f"DEBUG: Command = {' '.join(cmd)}")

    code, output = run_cmd(cmd)

    if debug:
        print(f"DEBUG: Return code = {code}")
        print(f"DEBUG: Output length = {len(output)} chars")
        if len(output) < 500:
            print(f"DEBUG: Output = {output}")
        else:
            print(f"DEBUG: Output (first 500 chars) = {output[:500]}")

    if code != 0:
        print(f"Error searching offers: {output}")
        return []

    try:
        offers = json.loads(output)
        if debug:
            print(f"DEBUG: Parsed {len(offers)} offers from API")

        # Filter in Python since vastai query syntax is unreliable
        filtered = []
        min_reliability_decimal = min_reliability / 100.0  # Convert % to decimal
        # Normalize excluded countries to lowercase for comparison
        excluded_lower = [c.lower() for c in exclude_countries]

        for i, offer in enumerate(offers):
            offer_gpus = offer.get("num_gpus", 0)
            offer_reliability = offer.get("reliability", 0)  # Already decimal (0.99 = 99%)
            offer_price = offer.get("dph_total", offer.get("dph_base", 999))
            offer_location = offer.get("geolocation", "")

            # Debug first few offers to see field values
            if debug and i < 3:
                print(f"DEBUG: Offer {i}: num_gpus={offer_gpus}, reliability={offer_reliability:.1%}, price=${offer_price:.2f}/hr, location={offer_location}")

            if offer_gpus != num_gpus:
                continue
            if offer_reliability < min_reliability_decimal:
                continue
            if max_price is not None and offer_price > max_price:
                continue
            # Check if country is excluded
            if excluded_lower and any(exc in offer_location.lower() for exc in excluded_lower):
                continue
            filtered.append(offer)

        if debug:
            print(f"DEBUG: After filtering: {len(filtered)} offers (num_gpus={num_gpus}, reliability>{min_reliability}, price<{max_price})")

        return filtered
    except json.JSONDecodeError:
        print(f"Error parsing offers: {output}")
        return []


def create_instance(offer_id: int, image: str, disk_gb: int = 50) -> tuple[bool, str]:
    """Create an instance from an offer."""
    cmd = [
        "vastai", "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--raw"
    ]

    code, output = run_cmd(cmd)
    if code != 0:
        return False, output

    try:
        result = json.loads(output)
        if result.get("success"):
            return True, str(result.get("new_contract"))
        else:
            return False, output
    except json.JSONDecodeError:
        # Sometimes it returns just the contract ID
        if output.isdigit():
            return True, output
        return False, output


def get_instance_info(contract_id: str, max_wait: int = 120) -> dict:
    """Wait for instance to be ready and get connection info."""
    print(f"Waiting for instance {contract_id} to be ready", end="", flush=True)

    start_time = time.time()
    while time.time() - start_time < max_wait:
        cmd = ["vastai", "show", "instance", contract_id, "--raw"]
        code, output = run_cmd(cmd)

        if code == 0:
            try:
                info = json.loads(output)
                status = info.get("actual_status", "unknown")

                if status == "running":
                    print(" Ready!")
                    return info
                elif status in ["loading", "starting", "created"]:
                    print(".", end="", flush=True)
                else:
                    print(f" Status: {status}")
            except json.JSONDecodeError:
                pass

        time.sleep(5)

    print(" Timeout!")
    return {}


def get_ssh_command(instance_info: dict) -> str:
    """Extract SSH command from instance info."""
    ssh_host = instance_info.get("ssh_host", "")
    ssh_port = instance_info.get("ssh_port", "")

    if ssh_host and ssh_port:
        return f"ssh -p {ssh_port} root@{ssh_host}"

    # Try public IP
    public_ip = instance_info.get("public_ipaddr", "")
    if public_ip and ssh_port:
        return f"ssh -p {ssh_port} root@{public_ip}"

    return "SSH info not available yet"


def main():
    parser = argparse.ArgumentParser(description="Find and rent cheapest GPU on vast.ai")
    parser.add_argument("--gpu", type=str, default="H100_SXM",
                        help="GPU model to search for (default: H100_SXM)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--image", type=str, default="aerdanielbryars101/lerobot-training:latest",
                        help="Docker image to use")
    parser.add_argument("--disk", type=int, default=50,
                        help="Disk space in GB (default: 50)")
    parser.add_argument("--min-reliability", type=float, default=95.0,
                        help="Minimum reliability %% (default: 95)")
    parser.add_argument("--max-price", type=float, default=5.0,
                        help="Maximum price $/hr (default: 5.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be rented without actually renting")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for instance to be ready")
    parser.add_argument("--exclude-countries", type=str, default="India,Taiwan,Thailand",
                        help="Comma-separated list of countries to exclude (default: India,Taiwan,Thailand)")

    args = parser.parse_args()

    # Parse excluded countries
    exclude_countries = [c.strip() for c in args.exclude_countries.split(",")] if args.exclude_countries else []

    print(f"Searching for cheapest {args.gpu} with {args.num_gpus} GPU(s) under ${args.max_price}/hr...")
    if exclude_countries:
        print(f"Excluding countries: {', '.join(exclude_countries)}")
    offers = search_offers(args.gpu, args.num_gpus, args.min_reliability, args.max_price, exclude_countries, debug=args.dry_run)

    if not offers:
        print(f"No {args.gpu} instances available!")
        sys.exit(1)

    # Get cheapest offer (already sorted by dph)
    cheapest = offers[0]
    offer_id = cheapest["id"]
    price = cheapest.get("dph_total", cheapest.get("dph_base", 0))
    gpu_name = cheapest.get("gpu_name", args.gpu)
    location = cheapest.get("geolocation", "Unknown")
    ram = cheapest.get("gpu_ram", 0) / 1024  # Convert to GB
    reliability = cheapest.get("reliability", 0)

    print()
    print("=" * 60)
    print("CHEAPEST OFFER FOUND")
    print("=" * 60)
    print(f"  Offer ID:    {offer_id}")
    print(f"  GPU:         {gpu_name}")
    print(f"  Price:       ${price:.4f}/hr")
    print(f"  VRAM:        {ram:.0f} GB")
    print(f"  Location:    {location}")
    print(f"  Reliability: {reliability*100:.1f}%")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        print(f"  vastai create instance {offer_id} --image {args.image} --disk {args.disk}")
        sys.exit(0)

    print(f"\nRenting instance with image: {args.image}")
    success, result = create_instance(offer_id, args.image, args.disk)

    if not success:
        print(f"Error creating instance: {result}")
        sys.exit(1)

    contract_id = result
    print(f"Instance created! Contract ID: {contract_id}")

    if args.no_wait:
        print(f"\nTo check status: vastai show instance {contract_id}")
        print(f"To destroy:      vastai destroy instance {contract_id}")
        sys.exit(0)

    # Wait for instance to be ready
    print()
    instance_info = get_instance_info(contract_id)

    if not instance_info:
        print(f"\nInstance not ready yet. Check manually:")
        print(f"  vastai show instance {contract_id}")
        sys.exit(1)

    ssh_cmd = get_ssh_command(instance_info)

    print()
    print("=" * 60)
    print("INSTANCE READY")
    print("=" * 60)
    print(f"  Contract ID: {contract_id}")
    print(f"  SSH Command: {ssh_cmd}")
    print()
    print("To run training:")
    print(f"  {ssh_cmd}")
    print("  # Then inside the container:")
    print("  python remote/train_remote.py --policy smolvla \\")
    print("      --dataset danbhf/sim_pick_place_40ep_rgbd_ee \\")
    print("      --from_pretrained lerobot/smolvla_base \\")
    print("      --cameras wrist_cam,overhead_cam \\")
    print("      --steps 20000 --batch_size 16 --eval_episodes 30")
    print()
    print(f"To destroy when done: vastai destroy instance {contract_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
