conda create -n picasso_pip_test python=3.10 -y
conda activate picasso_pip_test
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "picassosr"
picasso
conda deactivate