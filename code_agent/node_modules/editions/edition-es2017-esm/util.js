/**
 * Checks if a haystack string contains a needle string, used for ES5 compatibility.
 * @param haystack - The string to search within
 * @param needle - The string to search for
 * @returns True if the haystack does contain the needle, false if it does not contain the needle
 */
export function includes(haystack, needle) {
    return haystack.indexOf(needle) !== -1;
}
/**
 * Checks if a haystack string does not contain a needle string, used for ES5 compatibility.
 * @param haystack - The string to search within
 * @param needle - The string to search for
 * @returns True if the haystack does not contain the needle, false if it does contain the needle
 */
export function excludes(haystack, needle) {
    return haystack.indexOf(needle) === -1;
}
/**
 * Assert that the error is compatible with {@link ErrorLike}.
 * @param error - The error to assert
 * @throws {Error} If the error is not compatible with {@link ErrorLike}
 */
export function assertErrorLike(error) {
    if (!(error &&
        typeof error === 'object' &&
        'message' in error &&
        typeof error.message === 'string')) {
        console.error({ error: error });
        throw new Error('The error input is incompatible with ErrorLike');
    }
}
/**
 * Assert that the error is compatible with {@link ErrorInput}.
 * @param error - The error to assert for compatibility with {@link ErrorInput}
 * @throws {Error} If the error is not compatible with {@link ErrorInput}
 */
export function assertErrorInput(error) {
    if (typeof error === 'string' ||
        error instanceof Error ||
        (error &&
            typeof error === 'object' &&
            'message' in error &&
            typeof error.message === 'string')) {
        console.error({ error: error });
        throw new Error('The error input is incompatible with ErrorInput');
    }
}
export function detailedError(error, ...parents) {
    var _a, _b, _c, _d, _e, _f;
    let errorLike;
    if (typeof error === 'string') {
        errorLike = new Error(error);
        (_a = errorLike.code) !== null && _a !== void 0 ? _a : (errorLike.code = null);
        (_b = errorLike.level) !== null && _b !== void 0 ? _b : (errorLike.level = null);
    }
    else if (error instanceof Error) {
        errorLike = error;
        (_c = errorLike.code) !== null && _c !== void 0 ? _c : (errorLike.code = null);
        (_d = errorLike.level) !== null && _d !== void 0 ? _d : (errorLike.level = null);
    }
    else {
        assertErrorLike(error);
        errorLike = new Error(error.message);
        errorLike.code = (_e = error.code) !== null && _e !== void 0 ? _e : null;
        errorLike.level = (_f = error.level) !== null && _f !== void 0 ? _f : null;
    }
    // ensure parents are correct, note this is not all ancestors, just immediate parents
    if (errorLike.parent && parents.indexOf(errorLike.parent) === -1) {
        parents.push(errorLike.parent);
    }
    if (errorLike.parents && Array.isArray(errorLike.parents)) {
        parents.push(...errorLike.parents);
    }
    errorLike.parents = [];
    // error now has all the correct types
    const errorDetailed = errorLike;
    // ensure parents are valid, and not duplicated
    for (const parent of parents) {
        assertErrorLike(parent);
        if (errorDetailed.parents.indexOf(parent) === -1) {
            errorDetailed.parents.push(parent);
        }
    }
    // adjust the properties if necessary
    if (errorDetailed.code &&
        excludes(errorDetailed.message, `${errorDetailed.code}: `)) {
        errorDetailed.message = `${errorDetailed.code}: ${errorDetailed.message}`;
        errorDetailed.stack = `${errorDetailed.code}: ${errorDetailed.stack}`;
    }
    if (errorDetailed.level &&
        excludes(errorDetailed.message, `${errorDetailed.level}: `)) {
        errorDetailed.message = `${errorDetailed.level}: ${errorDetailed.message}`;
        errorDetailed.stack = `${errorDetailed.level}: ${errorDetailed.stack}`;
    }
    for (const parent of errorDetailed.parents) {
        if (excludes(errorDetailed.message, `\n↪ ${parent.message}`)) {
            errorDetailed.message = `${errorDetailed.message}\n↪ ${parent.message || parent}`;
            errorDetailed.stack = `${errorDetailed.stack}\n↪ ${parent.stack || parent.message || parent.parent}`;
        }
    }
    // return
    return errorDetailed;
}
