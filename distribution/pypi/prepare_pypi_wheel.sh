cd ../..
conda create -n picasso_pypi_wheel python=3.10
conda activate picasso_pypi_wheel
pip install build
pip install twine
rm -rf dist
rm -rf build
python -m build
twine check dist/*
conda deactivate
