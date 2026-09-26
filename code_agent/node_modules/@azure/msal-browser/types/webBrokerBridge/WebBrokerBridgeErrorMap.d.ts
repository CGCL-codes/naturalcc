import { AuthError } from "@azure/msal-common/browser";
import { WebBrokerBridgeError } from "./WebBrokerBridgeError.js";
/**
 * Lift a normalized `WebBrokerBridgeError` to one of the four base
 * MSAL error classes. Callers are responsible for lowering
 * transport-specific errors into `WebBrokerBridgeError` before calling
 * this function.
 */
export declare function toAuthError(err: WebBrokerBridgeError): AuthError;
//# sourceMappingURL=WebBrokerBridgeErrorMap.d.ts.map