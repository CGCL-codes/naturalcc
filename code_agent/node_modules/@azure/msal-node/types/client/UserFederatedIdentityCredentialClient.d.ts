import { AuthenticationResult, ClientConfiguration } from "@azure/msal-common/node";
import { BaseClient } from "./BaseClient.js";
import { CommonUserFederatedIdentityCredentialRequest } from "../request/CommonUserFederatedIdentityCredentialRequest.js";
/**
 * Client for the user_fic grant type (Leg 3 of Agent Identity).
 * Exchanges a federated identity credential (instance token) for a user-scoped token.
 * @internal
 */
export declare class UserFederatedIdentityCredentialClient extends BaseClient {
    constructor(configuration: ClientConfiguration);
    /**
     * Acquires a token using the user_fic grant type.
     * Always hits the network (no cache lookup for the network call).
     * Developers use acquireTokenSilent for cached FIC tokens.
     */
    acquireToken(request: CommonUserFederatedIdentityCredentialRequest): Promise<AuthenticationResult | null>;
    /**
     * Makes a network call to the token endpoint
     */
    private executeTokenRequest;
    /**
     * Builds the request body for the user_fic grant type
     */
    private createTokenRequestBody;
}
//# sourceMappingURL=UserFederatedIdentityCredentialClient.d.ts.map