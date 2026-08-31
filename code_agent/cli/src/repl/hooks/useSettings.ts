import { useState } from "react"

export function useSettings(){
    //settings
    const [files,setFiles] = useState<string[]>([])
    const [model,setModel] = useState<string>('deepseek/deepseek-chat')
    const [apiKey,setApiKey] = useState<string | null>(null)
    const [projectDir,setProjectDir] = useState<string>(process.cwd())
    const [symbol,setSymbol] = useState<string | null>(null)
    const [completionType,setCompletionType] = useState<string | null>(null)
    const [prefix,setPrefix] = useState<string>('')
    const [preview,setPreview] = useState<boolean>(false)
    const [feature,setFeature] = useState<string>('code_completion')
    const [featureConfig,setFeatureConfig] = useState<Record<string, unknown>>({})

    //Setting windows切换
    const [settingsWindow,setSettingsWindow] = useState<boolean>(true)

    return{
        files, model, apiKey, projectDir, symbol, completionType, prefix, preview, feature, featureConfig, settingsWindow,
        setFiles, setModel, setApiKey, setProjectDir, setSymbol, setCompletionType, setPrefix, setFeature, setFeatureConfig,
        togglePreview:() => setPreview(prev => !prev),
        toggleSettings:() => setSettingsWindow(prev => !prev),
        resetSettings:() => {
            setFiles([])
            setModel('deepseek/deepseek-chat')
            setApiKey(null)
            setProjectDir(process.cwd())
            setSymbol(null)
            setCompletionType(null)
            setPrefix('')
            setPreview(false)
            setFeature('code_completion')
            setFeatureConfig({})
        }
    }
}
