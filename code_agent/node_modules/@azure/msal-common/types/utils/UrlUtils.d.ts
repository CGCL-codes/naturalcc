import { AuthorizeResponse } from "../response/AuthorizeResponse.js";
import { Logger } from "../logger/Logger.js";
/**
 * Parses hash string from given string. Returns empty string if no hash symbol is found.
 * @param hashString
 */
export declare function stripLeadingHashOrQuery(responseString: string): string;
/**
 * Returns URL hash as server auth code response object.
 */
export declare function getDeserializedResponse(responseString: string): AuthorizeResponse | null;
/**
 * Utility to create a URL from the params map
 */
export declare function mapToQueryString(parameters: Map<string, string>): string;
/**
 * Normalizes URLs for comparison per MDN & RFC 3986 standards:
 * - Hash/fragment is removed
 * - Scheme and host are lowercased (case-insensitive per spec)
 * - Path and query parameters preserve original casing (case-sensitive per spec)
 * - Percent-encoding in pathname is normalized (e.g., %27 and ' are treated equivalently)
 * - Ensures pathname ends with /
 * Throws a urlParseError if the provided URL is malformed and cannot be parsed.
 * @param url - URL to normalize
 * @param logger - Optional logger used to log parse failures
 * @param correlationId - Optional correlationId associated with the log entry
 * @returns Normalized URL string for comparison
 */
export declare function normalizeUrlForComparison(url: string, logger?: Logger, correlationId?: string): string;
/**
 * Validates that the provided value is a well-formed, parseable absolute URL.
 * Throws a urlParseError if the value cannot be parsed by the URL API (e.g. the
 * literal string "null", an empty string, or any malformed/relative URL). Use this
 * to guard against persisting an invalid value (such as a redirect URL) to the cache.
 * @param url - URL to validate
 * @param logger - Optional logger used to log validation failures
 * @param correlationId - Optional correlationId associated with the log entry
 */
export declare function validateUrl(url: string, logger?: Logger, correlationId?: string): void;
//# sourceMappingURL=UrlUtils.d.ts.map