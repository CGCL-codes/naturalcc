import { ResetPasswordApiClient } from "./ResetPasswordApiClient.js";
import { SignupApiClient } from "./SignupApiClient.js";
import { SignInApiClient } from "./SignInApiClient.js";
import { RegisterApiClient } from "./RegisterApiClient.js";
import { ICustomAuthApiClient } from "./ICustomAuthApiClient.js";
import { IHttpClient } from "../http_client/IHttpClient.js";
import { Logger } from "@azure/msal-common/browser";
import { CustomAuthRequestInterceptor } from "../../../configuration/CustomAuthRequestInterceptor.js";
export declare class CustomAuthApiClient implements ICustomAuthApiClient {
    signInApi: SignInApiClient;
    signUpApi: SignupApiClient;
    resetPasswordApi: ResetPasswordApiClient;
    registerApi: RegisterApiClient;
    constructor(customAuthApiBaseUrl: string, clientId: string, httpClient: IHttpClient, capabilities?: string, customAuthApiQueryParams?: Record<string, string>, requestInterceptor?: CustomAuthRequestInterceptor, logger?: Logger);
}
//# sourceMappingURL=CustomAuthApiClient.d.ts.map