/**
 * Checks if a haystack string contains a needle string, used for ES5 compatibility.
 * @param haystack - The string to search within
 * @param needle - The string to search for
 * @returns True if the haystack does contain the needle, false if it does not contain the needle
 */
export function includes(haystack: string, needle: string): boolean {
	return haystack.indexOf(needle) !== -1
}

/**
 * Checks if a haystack string does not contain a needle string, used for ES5 compatibility.
 * @param haystack - The string to search within
 * @param needle - The string to search for
 * @returns True if the haystack does not contain the needle, false if it does contain the needle
 */
export function excludes(haystack: string, needle: string): boolean {
	return haystack.indexOf(needle) === -1
}

// ============================================================================
// ErrorLike

/** The {@link Error}-like properties that provide details to {@link ErrorDetailed} */
export interface ErrorLike {
	/** The message that describes the error */
	message: string
	/** The code to identify the category of the error for automated processing */
	code?: unknown
	/** The severity level of the error */
	level?: unknown
	/** The parent of the error */
	parent?: unknown
	/** The parents of the error */
	parents?: unknown
	/** The stack of the error, this is used internally. */
	stack?: string
}

/**
 * Assert that the error is compatible with {@link ErrorLike}.
 * @param error - The error to assert
 * @throws {Error} If the error is not compatible with {@link ErrorLike}
 */
export function assertErrorLike(error: unknown): asserts error is ErrorLike {
	if (
		!(
			error &&
			typeof error === 'object' &&
			'message' in error &&
			typeof error.message === 'string'
		)
	) {
		console.error({ error: error })
		throw new Error('The error input is incompatible with ErrorLike')
	}
}

// ============================================================================
// ErrorInput

/** The range of compatible error inputs for {@link ErrorDetailed} */
export type ErrorInput = ErrorLike | Error | string

/**
 * Assert that the error is compatible with {@link ErrorInput}.
 * @param error - The error to assert for compatibility with {@link ErrorInput}
 * @throws {Error} If the error is not compatible with {@link ErrorInput}
 */
export function assertErrorInput(error: unknown): asserts error is ErrorInput {
	if (
		typeof error === 'string' ||
		error instanceof Error ||
		(error &&
			typeof error === 'object' &&
			'message' in error &&
			typeof error.message === 'string')
	) {
		console.error({ error: error })
		throw new Error('The error input is incompatible with ErrorInput')
	}
}

// ============================================================================
// ErrorDetailed

/** The resultant detailed error instance created by {@link detailedError} */
export interface ErrorDetailed extends ErrorLike, Error {
	code: string | number | null
	level: string | number | null
	parents: ErrorLike[]
}

/* eslint-disable @typescript-eslint/unified-signatures -- this eslint rule prevents the types from actually being correct on the callers, such as giving correct details for message when you mouse over it */
// disabling that rule, and doing what we do, it enables intellisense for message:
// try {} catch (error: unknown) { throw detailedError({ message: '...', code: '...' }, error as ErrorInput) }
// enabling the rule and just doing unknown, requires the caller to typecast for such intellisense:
// try {} catch (error: unknown) { throw detailedError({ message: '...', code: '...' } as ErrorLike, error as ErrorInput) }

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
export function detailedError(
	error: ErrorLike,
	...parents: unknown[]
): ErrorDetailed
export function detailedError(
	error: ErrorInput,
	...parents: unknown[]
): ErrorDetailed
export function detailedError(
	error: unknown,
	...parents: unknown[]
): ErrorDetailed
export function detailedError<T>(
	error: T extends ErrorInput ? ErrorInput : unknown,
	...parents: unknown[]
): ErrorDetailed {
	let errorLike: Error & ErrorLike
	if (typeof error === 'string') {
		errorLike = new Error(error)
		errorLike.code ??= null
		errorLike.level ??= null
	} else if (error instanceof Error) {
		errorLike = error as Error & ErrorLike
		errorLike.code ??= null
		errorLike.level ??= null
	} else {
		assertErrorLike(error)
		errorLike = new Error(error.message)
		errorLike.code = error.code ?? null
		errorLike.level = error.level ?? null
	}

	// ensure parents are correct, note this is not all ancestors, just immediate parents
	if (errorLike.parent && parents.indexOf(errorLike.parent) === -1) {
		parents.push(errorLike.parent)
	}
	if (errorLike.parents && Array.isArray(errorLike.parents)) {
		parents.push(...(errorLike.parents as unknown[]))
	}
	errorLike.parents = []

	// error now has all the correct types
	const errorDetailed = errorLike as ErrorDetailed

	// ensure parents are valid, and not duplicated
	for (const parent of parents) {
		assertErrorLike(parent)
		if (errorDetailed.parents.indexOf(parent) === -1) {
			errorDetailed.parents.push(parent)
		}
	}

	// adjust the properties if necessary
	if (
		errorDetailed.code &&
		excludes(errorDetailed.message, `${errorDetailed.code}: `)
	) {
		errorDetailed.message = `${errorDetailed.code}: ${errorDetailed.message}`
		errorDetailed.stack = `${errorDetailed.code}: ${errorDetailed.stack}`
	}
	if (
		errorDetailed.level &&
		excludes(errorDetailed.message, `${errorDetailed.level}: `)
	) {
		errorDetailed.message = `${errorDetailed.level}: ${errorDetailed.message}`
		errorDetailed.stack = `${errorDetailed.level}: ${errorDetailed.stack}`
	}
	for (const parent of errorDetailed.parents) {
		if (excludes(errorDetailed.message, `\n↪ ${parent.message}`)) {
			errorDetailed.message = `${errorDetailed.message}\n↪ ${parent.message || parent}`
			errorDetailed.stack = `${errorDetailed.stack}\n↪ ${parent.stack || parent.message || parent.parent}`
		}
	}

	// return
	return errorDetailed
}
