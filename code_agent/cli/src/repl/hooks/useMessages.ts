import { useState } from "react";
import type { Message, Role } from "../../types/message";
import { now } from "../utils.js";

export function useMessages(){
    const [messages, setMessages] = useState<Message[]>([])

    function msg(role: Role, content: string): Message {
        return { role, content, time: now() }
    }

    return{
        messages,
        addMsg(role : Role, content : string) : void{
            setMessages(prev => [...prev, msg(role, content)])
        },
        clearMessages:() => {
            setMessages([])
        }
    }
}