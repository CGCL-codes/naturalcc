/**
 * postMessage envelope type used to relay an auth response from a popup-relay
 * page (see `runPopupRelay`) back to the embedded frame that opened it. See
 * {@link BrowserAuthOptions.popupRelayUri}.
 *
 * Kept dependency-free so both the relay-page entry (`./index.js`) and the
 * client-side helpers (`./relayClient.js`) can import it without either bundle
 * pulling in the other's dependencies.
 *
 * @internal
 */
export declare const POPUP_RELAY_RESPONSE_TYPE = "msal:popup-relay-response:v1";
//# sourceMappingURL=constants.d.ts.map