import { Logger } from "@azure/msal-common/browser";
/**
 * Filters the headers returned by a {@link CustomAuthRequestInterceptor},
 * keeping only those that conform to the custom-auth header naming rules.
 *
 * Rules (mirrors the iOS / Android native auth implementations):
 *  - Header names must start with `x-` (case-insensitive); others are dropped.
 *  - Header names that start with any reserved prefix (`x-client-`, `x-ms-`,
 *    `x-broker-`, `x-app-`) are dropped.
 *  - Headers with empty/whitespace-only names or null/undefined values are dropped.
 *
 * Dropped headers are logged as warnings (PII-safe) when a logger is provided.
 *
 * @param headers - Raw headers returned by the interceptor.
 * @param logger - Optional logger used to emit warnings for dropped headers.
 * @param correlationId - Optional correlation id forwarded to the logger.
 * @returns A new record containing only the headers that pass the filter,
 *          preserving the original casing of header names.
 */
export declare function filterCustomHeaders(headers: Record<string, string> | null | undefined, logger?: Logger, correlationId?: string): Record<string, string>;
//# sourceMappingURL=CustomHeaderUtils.d.ts.map