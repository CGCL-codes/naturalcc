/**
 * Checks if a haystack string contains a needle string, used for ES5 compatibility.
 * @param haystack - The string to search within
 * @param needle - The string to search for
 * @returns True if the haystack does contain the needle, false if it does not contain the needle
 */
export declare function includes(haystack: string, needle: string): boolean;
/**
 * Checks if a haystack string does not contain a needle string, used for ES5 compatibility.
 * @param haystack - The string to search within
 * @param needle - The string to search for
 * @returns True if the haystack does not contain the needle, false if it does contain the needle
 */
export declare function excludes(haystack: string, needle: string): boolean;
/** The {@link Error}-like properties that provide details to {@link ErrorDetailed} */
export interface ErrorLike {
    /** The message that describes the error */
    message: string;
    /** The code to identify the category of the error for automated processing */
    code?: unknown;
    /** The severity level of the error */
    level?: unknown;
    /** The parent of the error */
    parent?: unknown;
    /** The parents of the error */
    parents?: unknown;
    /** The stack of the error, this is used internally. */
    stack?: string;
}
/**
 * Assert that the error is compatible with {@link ErrorLike}.
 * @param error - The error to assert
 * @throws {Error} If the error is not compatible with {@link ErrorLike}
 */
export declare function assertErrorLike(error: unknown): asserts error is ErrorLike;
/** The range of compatible error inputs for {@link ErrorDetailed} */
export type ErrorInput = ErrorLike | Error | string;
/**
 * Assert that the error is compatible with {@link ErrorInput}.
 * @param error - The error to assert for compatibility with {@link ErrorInput}
 * @throws {Error} If the error is not compatible with {@link ErrorInput}
 */
export declare function assertErrorInput(error: unknown): asserts error is ErrorInput;
/** The resultant detailed error instance created by {@link detailedError} */
export interface ErrorDetailed extends ErrorLike, Error {
    code: string | number | null;
    level: string | number | null;
    parents: ErrorLike[];
}
/**
 * Ensure the error is a proper error instance, and when stringified include any code, level, and parent details if defined upon construction.
 * We do this instead of a class extension, as class extensions do not interop well on Node.js 0.8, which is a target.
 * @param error - The error, or its details, accepts {@link ErrorInput}, ideally {@link ErrorLike}
 * @param parents - The parent(s) of the error, accepts {@link ErrorLike}
 * @returns The detailed error instance, matching {@link ErrorDetailed}
 * @throws {Error} If the error or parent were incompatible with {@link ErrorLike}
 * @example
 * ```ts
 * try {} catch (error: unknown) { throw detailedError({ message: '...', code: '...' } as ErrorLike, error as ErrorInput) }
 * ```
 */
export declare function detailedError(error: ErrorLike, ...parents: unknown[]): ErrorDetailed;
export declare function detailedError(error: ErrorInput, ...parents: unknown[]): ErrorDetailed;
export declare function detailedError(error: unknown, ...parents: unknown[]): ErrorDetailed;
//# sourceMappingURL=util.d.ts.map