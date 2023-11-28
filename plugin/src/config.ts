import { workspace } from 'vscode'
import { pluginName } from './extension'
export class NccConfig {
    public server: string
    public debounce: Number
    public maxLength: Number
    public topK: Number
    public topP: Number
    public temperature: Number
    constructor() {
        this.server = workspace.getConfiguration(pluginName).get('server') as string
        this.debounce = workspace.getConfiguration(pluginName).get('debounce') as Number
        this.maxLength = workspace.getConfiguration(pluginName).get('maxLength') as Number
        this.topK = workspace.getConfiguration(pluginName).get('topK') as Number
        this.topP = workspace.getConfiguration(pluginName).get('topP') as Number
        this.temperature = workspace.getConfiguration(pluginName).get('temperature') as Number
    }
}

function updateConfig(c: NccConfig) {
    c.server = workspace.getConfiguration(pluginName).get('server') as string
    c.debounce = workspace.getConfiguration(pluginName).get('debounce') as Number
    c.maxLength = workspace.getConfiguration(pluginName).get('maxLength') as Number
    c.topK = workspace.getConfiguration(pluginName).get('topK') as Number
    c.topP = workspace.getConfiguration(pluginName).get('topP') as Number
    c.temperature = workspace.getConfiguration(pluginName).get('temperature') as Number
}

var config: NccConfig

export function onConfigChange() {
    if(!config) {
        config = new NccConfig()
    }
    updateConfig(config)
}

export function getConfig(): NccConfig {
    onConfigChange()
    return config;
}

