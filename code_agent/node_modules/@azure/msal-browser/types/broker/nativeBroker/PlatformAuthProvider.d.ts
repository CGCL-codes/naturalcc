import { LoggerOptions, IPerformanceClient, Logger, Constants } from "@azure/msal-common/browser";
import { BrowserConfiguration } from "../../config/Configuration.js";
import { IPlatformAuthHandler } from "./IPlatformAuthHandler.js";
/**
 * Checks if the platform broker is available in the current environment.
 * @param domConfig - Whether to enable platform broker DOM API support (required)
 * @param loggerOptions - Optional logger options
 * @param perfClient - Optional performance client
 * @param correlationId - Optional correlation ID
 * @returns Promise<boolean> indicating if platform broker is available
 */
export declare function isPlatformBrokerAvailable(domConfig: boolean, loggerOptions?: LoggerOptions, perfClient?: IPerformanceClient, correlationId?: string): Promise<boolean>;
export declare function getPlatformAuthProvider(logger: Logger, performanceClient: IPerformanceClient, correlationId: string, nativeBrokerHandshakeTimeout?: number, enablePlatformBrokerDOMSupport?: boolean): Promise<IPlatformAuthHandler | undefined>;
/**
 * Returns boolean indicating whether or not the request should attempt to use platform broker
 * @param logger
 * @param config
 * @param correlationId
 * @param platformAuthProvider
 * @param authenticationScheme
 */
export declare function isPlatformAuthAllowed(config: BrowserConfiguration, logger: Logger, correlationId: string, platformAuthProvider?: IPlatformAuthHandler, authenticationScheme?: Constants.AuthenticationScheme): boolean;
//# sourceMappingURL=PlatformAuthProvider.d.ts.map