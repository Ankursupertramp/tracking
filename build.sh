#!/bin/bash
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.11.7
pyenv global 3.11.7
python -m pip install --upgrade pip
pip install -r requirements.txt
