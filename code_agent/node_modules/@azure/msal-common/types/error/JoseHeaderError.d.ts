import { AuthError } from "./AuthError.js";
import * as JoseHeaderErrorCodes from "./JoseHeaderErrorCodes.js";
export { JoseHeaderErrorCodes };
/**
 * Error thrown when there is an error in the client code running on the browser.
 */
export declare class JoseHeaderError extends AuthError {
    constructor(errorCode: string, correlationId: string, errorMessage?: string);
}
/** Returns JoseHeaderError object */
export declare function createJoseHeaderError(code: string, correlationId: string): JoseHeaderError;
//# sourceMappingURL=JoseHeaderError.d.ts.map