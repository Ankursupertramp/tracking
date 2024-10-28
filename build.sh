#!/bin/bash

# Ensure pyenv is installed
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# Set up pyenv environment
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install and use Python 3.11.7
pyenv install -s 3.11.7
pyenv global 3.11.7

# Upgrade pip and install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# If distutils is not included in the Python installation, install it manually
if ! python -c "import distutils" &>/dev/null; then
    pip install distutils
fi
