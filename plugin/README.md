# Ncc Completor

## Enable / Disable

To enable the plugin: 
1. Press `ctrl + shift + p` or `cmd + shift + p`.
2. Type in `Enable nccc`.

To disable the plugin: 
1. Press `ctrl + shift + p` or `cmd + shift + p`.
2. Type in `Disable nccc`.

## Configurations

To set the arguments:

1. Press `ctrl + shift + p` or `cmd + shift + p`.
2. Type in `Preferences: Open Settings (UI)` or `Preferences: Open User Settings (JSON)`.
3. Search for `nccc`.
4. Modify settings you are interested in.

All the arguments are listed below:

* `nccc.server`: The server running the model. default: "http://127.0.0.1:5000"
* `nccc.debounce`: The inter-trigger time(in ms) to wait between completion requests. default: 1500
* `nccc.maxLength`: The maximium length of generation. default: 200
* `nccc.topK`: The top_k used in generation. default: 10
* `nccc.topP`: The top_p used in generation. default: 0.95
* `nccc.temperature`: The temperature used in generation. default: 0.1
* `nccc.contextLineCount`: The lines to consider(i.e. pass to the model) when requesting completion. default: 5