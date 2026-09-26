import { TokenClaims } from "./TokenClaims.js";
/**
 * Extract token by decoding the rawToken
 *
 * @param encodedToken
 */
export declare function extractTokenClaims(encodedToken: string, base64Decode: (input: string) => string, correlationId: string): TokenClaims;
/**
 * Check if the signin_state claim contains "kmsi"
 * @param idTokenClaims
 * @returns
 */
export declare function isKmsi(idTokenClaims: TokenClaims): boolean;
/**
 * decode a JWT
 *
 * @param authToken
 */
export declare function getJWSPayload(authToken: string, correlationId: string): string;
//# sourceMappingURL=AuthToken.d.ts.map