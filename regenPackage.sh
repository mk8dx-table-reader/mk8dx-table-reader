rm -rf dist build *.egg-info
python setup.py sdist bdist_wheel
pip uninstall mk8dx_table_reader -y
pip install dist/mk8dx_table_reader-*py3-none-any.whl 

# python -m twine upload dist/*