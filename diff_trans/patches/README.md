This folder contains patches for fixing libraries

- `solver.py`

Replaces `while` loop used in mjx's solver with `scan` to preserve differentiability

```bash
ln ./solver.py "<PYTHON_LIB_FOLDER>/site-packages/mujoco/mjx/_src/solver.py"
```