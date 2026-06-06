import numpy, subprocess, glob, os
print('numpy version:', numpy.__version__)
so = None
for d in ['core', '_core']:
    for f in glob.glob(os.path.join(os.path.dirname(numpy.__file__), d, '_multiarray_umath*.so')):
        so = f
        break
    if so:
        break
print('numpy so:', so)
if so:
    r = subprocess.run(['nm', '-D', so], capture_output=True, text=True)
    for sym in ['npy_exp', '__svml_exp8', 'npy_log', 'cblas_sdot64_', 'cblas_sdot']:
        found = any(sym in line for line in r.stdout.splitlines())
        print('  ' + sym + ': ' + ('found' if found else 'NOT found'))
else:
    print('  numpy so NOT found in expected locations')
