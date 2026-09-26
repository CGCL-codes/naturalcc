/**
 * Unified error taxonomy shared by NAA and PWB.
 *
 * Each code maps to exactly one of the four base MSAL error classes
 * (`AuthError`, `ClientAuthError`, `ServerError`,
 * `InteractionRequiredAuthError`) via `toAuthError` in
 * `./WebBrokerBridgeErrorMap`.
 */
export declare const WebBrokerBridgeErrorCode: {
    readonly UserInteractionRequired: "user_interaction_required";
    readonly UserCanceled: "user_canceled";
    readonly NoNetwork: "no_network";
    readonly AccountUnavailable: "account_unavailable";
    readonly BridgeDisabled: "bridge_disabled";
    readonly BridgeUnavailable: "bridge_unavailable";
    readonly TransientError: "transient_error";
    readonly PersistentError: "persistent_error";
    readonly BridgeTimeout: "bridge_timeout";
    readonly BridgeHandshakeFailed: "bridge_handshake_failed";
    readonly BridgeConnectionReset: "bridge_connection_reset";
    readonly BridgeResponseInvalid: "bridge_response_invalid";
    readonly PopupWillRedirect: "popup_will_redirect";
    readonly ShrGenerationError: "shr_generation_error";
    readonly Unknown: "unknown";
};
export type WebBrokerBridgeErrorCode = (typeof WebBrokerBridgeErrorCode)[keyof typeof WebBrokerBridgeErrorCode];
/**
 * Normalized error payload carried between the broker and the embedded
 * app. Bridge-specific transports (NAA `BridgeError`, PWB `errorPayload`)
 * are lowered into this shape before being lifted to an MSAL error class.
 */
export interface WebBrokerBridgeError {
    /**
     * Common taxonomy code — see `WebBrokerBridgeErrorCode`.
     */
    readonly code: WebBrokerBridgeErrorCode;
    /**
     * Underlying provider-specific error code preserved for classes that
     * accept a variable code (e.g. `invalid_grant` on `ServerError`).
     * Ignored for codes whose mapping is a fixed constant.
     */
    readonly innerErrorCode?: string;
    /** Server sub-error (e.g. `consent_required`). */
    readonly subError?: string;
    /** Human-readable description forwarded from the underlying response. */
    readonly description?: string;
    /** Correlation id, when available. */
    readonly correlationId?: string;
}
//# sourceMappingURL=WebBrokerBridgeError.d.ts.map