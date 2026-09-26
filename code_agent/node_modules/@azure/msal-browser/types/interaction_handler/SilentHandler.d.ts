import { Logger, IPerformanceClient, Authority, CommonAuthorizationUrlRequest } from "@azure/msal-common/browser";
import { BrowserConfiguration } from "../config/Configuration.js";
/**
 * Navigates the given hidden iframe to the provided URL.
 * @param frame
 * @param requestUrl
 */
export declare function initiateCodeRequest(frame: HTMLIFrameElement, requestUrl: string, logger: Logger, correlationId: string): Promise<HTMLIFrameElement>;
export declare function initiateCodeFlowWithPost(frame: HTMLIFrameElement, config: BrowserConfiguration, authority: Authority, request: CommonAuthorizationUrlRequest, logger: Logger, performanceClient: IPerformanceClient): Promise<HTMLIFrameElement>;
export declare function initiateEarRequest(frame: HTMLIFrameElement, config: BrowserConfiguration, authority: Authority, request: CommonAuthorizationUrlRequest, logger: Logger, performanceClient: IPerformanceClient): Promise<HTMLIFrameElement>;
/**
 * @hidden
 * Creates a new hidden iframe for silent token renewal. Callers navigate it
 * (set `src` or submit a form) after registering the response listener.
 * @ignore
 */
export declare function createHiddenIframe(): HTMLIFrameElement;
/**
 * @hidden
 * Removes a hidden iframe from `document.body` if it is a direct child.
 * @param iframe - The iframe element to remove.
 */
export declare function removeHiddenIframe(iframe: HTMLIFrameElement): void;
//# sourceMappingURL=SilentHandler.d.ts.map