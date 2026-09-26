import { IHttpClient } from "../http_client/IHttpClient.js";
import { Logger, ServerTelemetryManager } from "@azure/msal-common/browser";
import { CustomAuthRequestInterceptor } from "../../../configuration/CustomAuthRequestInterceptor.js";
export declare abstract class BaseApiClient {
    private readonly clientId;
    private httpClient;
    private customAuthApiQueryParams?;
    private requestInterceptor?;
    private logger?;
    private readonly baseRequestUrl;
    constructor(baseUrl: string, clientId: string, httpClient: IHttpClient, customAuthApiQueryParams?: Record<string, string> | undefined, requestInterceptor?: CustomAuthRequestInterceptor | undefined, logger?: Logger | undefined);
    protected request<T>(endpoint: string, data: Record<string, string | boolean>, telemetryManager: ServerTelemetryManager, correlationId: string): Promise<T>;
    protected ensureContinuationTokenIsValid(continuationToken: string | undefined, correlationId: string): void;
    private readResponseCorrelationId;
    private getCommonHeaders;
    private handleApiResponse;
    private getAdditionalHeaders;
}
//# sourceMappingURL=BaseApiClient.d.ts.map