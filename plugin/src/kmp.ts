
// use KMP to find the overlap between the suffix of user input and prefix of completion result
export function getOverlap(text: string, pattern: string): number {

    text = ' ' + text
    pattern = ' ' + pattern

    let nxt = new Array(pattern.length)
    nxt[1] = 0
    for(let i = 2, j = 0; i < pattern.length; i++) {
        while(j > 0 && pattern[i] != pattern[j+ 1]) {
            j = nxt[j]
        }
        if(pattern[i] == pattern[j + 1]) {
            j++;
        }
        nxt[i] = j;
    }
    
    let ret = 0;
    for(let i = 1, j = 0; i < text.length; i++) {
        while(j > 0 && (j == pattern.length - 1 || text[i] != pattern[j + 1])) {
            j = nxt[j]
        }
        if(text[i] == pattern[j + 1]) {
            j++
        }
        ret = j
    }
    return ret;
}

