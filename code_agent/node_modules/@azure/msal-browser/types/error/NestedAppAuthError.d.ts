import { AuthError } from "@azure/msal-common/browser";
export declare class NestedAppAuthError extends AuthError {
    constructor(errorCode: string, correlationId: string, errorMessage?: string);
    static createUnsupportedError(correlationId?: string): NestedAppAuthError;
}
//# sourceMappingURL=NestedAppAuthError.d.ts.map