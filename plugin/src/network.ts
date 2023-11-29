import axios from 'axios';
import { NccConfig, getConfig } from './config';

class NccResult {
    public generated_text: string;
    constructor(text: string) {
        this.generated_text = text
    }
}

class NccPrompt {
    public config: NccConfig
    public prompt: string
    constructor(prompt: string) {
        this.config = getConfig()
        this.prompt = prompt
    }

    async fire(): Promise<NccResult> {
        try {
            const response = await axios.post(encodeURI(this.config.server + '/complete'), this)
            return response.data as NccResult
        } catch(err) {
            console.log(err)
        }
        return new NccResult("")
    }
}

export async function getCompletion(prompt: string): Promise<string> {
    const nccPrompt = new NccPrompt(prompt)
    const result = await nccPrompt.fire()
    return result.generated_text
}