"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.patchNLS = patchNLS;
const regex = /^%([\w\d.]+)%$/i;
function createPatcher(translations) {
    return (value) => {
        if (typeof value !== 'string') {
            return value;
        }
        const match = regex.exec(value);
        if (!match) {
            return value;
        }
        const translation = translations[match[1]];
        if (translation === undefined) {
            throw new Error(`No translation found for ${value}`);
        }
        return translation;
    };
}
function patchNLS(manifest, translations) {
    const patcher = createPatcher(translations);
    return JSON.parse(JSON.stringify(manifest, (_, value) => patcher(value)));
}
//# sourceMappingURL=nls.js.map