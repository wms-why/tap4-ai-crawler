unset http_proxy
unset https_proxy
proxychains zsh -c ".venv/bin/python main_api.py"