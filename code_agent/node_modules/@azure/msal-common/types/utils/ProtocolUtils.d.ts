import { ICrypto } from "../crypto/ICrypto.js";
import { RequestStateObject } from "./StateTypes.js";
/**
 * Appends user state with random guid, or returns random guid.
 * @param cryptoObj
 * @param userState
 * @param meta
 * @param correlationId
 */
export declare function setRequestState(cryptoObj: ICrypto, userState: string | undefined, meta: Record<string, string> | undefined, correlationId: string): string;
/**
 * Generates the state value used by the common library.
 * @param cryptoObj
 * @param correlationId
 * @param meta
 */
export declare function generateLibraryState(cryptoObj: ICrypto, correlationId: string, meta?: Record<string, string>): string;
/**
 * Parses the state into the RequestStateObject, which contains the LibraryState info and the state passed by the user.
 * @param base64Decode
 * @param state
 * @param correlationId
 */
export declare function parseRequestState(base64Decode: (input: string) => string, state: string, correlationId: string): RequestStateObject;
//# sourceMappingURL=ProtocolUtils.d.ts.map