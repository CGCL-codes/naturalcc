import { IPerformanceClient, Logger } from "@azure/msal-common/browser";
import type { WaitForBridgeRequest } from "../utils/BrowserUtils.js";
/**
 * The navigation the popup-relay page should perform in its IdP child popup:
 * either a GET navigation to a URL, or a POST form submission (used by the
 * form_post and EAR response modes).
 *
 * @internal
 */
export type PopupRelayAction = {
    method: "GET";
    url: string;
} | {
    method: "POST";
    action: string;
    fields: Record<string, string>;
};
/**
 * Builds the URL of the popup-relay page to open. The action the relay page
 * must perform (GET navigation or POST form) is carried in the page's hash
 * (client-side only, never sent to a server), keyed by the per-request
 * library-state `id` so the relay page can listen on the right BroadcastChannel
 * and echo the id back. The relay URI is resolved against the app origin, so it
 * must be same-origin as the embedded frame. See `runPopupRelay`.
 *
 * @internal
 */
export declare function buildPopupRelayUrl(popupRelayUri: string, id: string, action: PopupRelayAction, correlationId: string): string;
/**
 * Waits for the popup-relay page to relay the raw auth response back to the
 * embedded frame via postMessage. Used in place of `waitForBridgeResponse`
 * when `auth.popupRelayUri` is configured, because third-party storage
 * partitioning blocks the BroadcastChannel the redirect bridge uses from
 * reaching the embedded frame.
 *
 * The response is accepted only from the popup window we opened, on our own
 * origin (the relay page is same-origin as the embedded frame), and only when
 * it carries the matching per-request library-state id.
 *
 * @internal
 */
export declare function waitForPopupRelayResponse(timeoutMs: number, logger: Logger, request: WaitForBridgeRequest, popupWindow: Window, performanceClient: IPerformanceClient): Promise<string>;
//# sourceMappingURL=relayClient.d.ts.map