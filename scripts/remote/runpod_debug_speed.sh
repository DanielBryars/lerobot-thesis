# Check create_tasks and other slow operations
source /root/octo_env/bin/activate
echo "=== octo_model_pt.py create_tasks ==="
grep -n "create_tasks" /root/octo-pytorch/octo/model/octo_model_pt.py | head -5; echo "GREP_DONE"
sed -n '540,600p' /root/octo-pytorch/octo/model/octo_model_pt.py 2>&1; echo "CREATE_TASKS_DONE"
echo "=== Check _verify_shapes overhead ==="
grep -n "_verify_shapes" /root/octo-pytorch/octo/model/octo_model_pt.py | head -5; echo "VERIFY_DONE"
