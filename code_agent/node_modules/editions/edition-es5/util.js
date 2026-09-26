"use strict";
var __read = (this && this.__read) || function (o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
var __values = (this && this.__values) || function(o) {
    var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
    if (m) return m.call(o);
    if (o && typeof o.length === "number") return {
        next: function () {
            if (o && i >= o.length) o = void 0;
            return { value: o && o[i++], done: !o };
        }
    };
    throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.includes = includes;
exports.excludes = excludes;
exports.assertErrorLike = assertErrorLike;
exports.assertErrorInput = assertErrorInput;
exports.detailedError = detailedError;
/**
 * Checks if a haystack string contains a needle string, used for ES5 compatibility.
 * @param haystack - The string to search within
 * @param needle - The string to search for
 * @returns True if the haystack does contain the needle, false if it does not contain the needle
 */
function includes(haystack, needle) {
    return haystack.indexOf(needle) !== -1;
}
/**
 * Checks if a haystack string does not contain a needle string, used for ES5 compatibility.
 * @param haystack - The string to search within
 * @param needle - The string to search for
 * @returns True if the haystack does not contain the needle, false if it does contain the needle
 */
function excludes(haystack, needle) {
    return haystack.indexOf(needle) === -1;
}
/**
 * Assert that the error is compatible with {@link ErrorLike}.
 * @param error - The error to assert
 * @throws {Error} If the error is not compatible with {@link ErrorLike}
 */
function assertErrorLike(error) {
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
function assertErrorInput(error) {
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
function detailedError(error) {
    var e_1, _a, e_2, _b;
    var _c, _d, _e, _f, _g, _h;
    var parents = [];
    for (var _i = 1; _i < arguments.length; _i++) {
        parents[_i - 1] = arguments[_i];
    }
    var errorLike;
    if (typeof error === 'string') {
        errorLike = new Error(error);
        (_c = errorLike.code) !== null && _c !== void 0 ? _c : (errorLike.code = null);
        (_d = errorLike.level) !== null && _d !== void 0 ? _d : (errorLike.level = null);
    }
    else if (error instanceof Error) {
        errorLike = error;
        (_e = errorLike.code) !== null && _e !== void 0 ? _e : (errorLike.code = null);
        (_f = errorLike.level) !== null && _f !== void 0 ? _f : (errorLike.level = null);
    }
    else {
        assertErrorLike(error);
        errorLike = new Error(error.message);
        errorLike.code = (_g = error.code) !== null && _g !== void 0 ? _g : null;
        errorLike.level = (_h = error.level) !== null && _h !== void 0 ? _h : null;
    }
    // ensure parents are correct, note this is not all ancestors, just immediate parents
    if (errorLike.parent && parents.indexOf(errorLike.parent) === -1) {
        parents.push(errorLike.parent);
    }
    if (errorLike.parents && Array.isArray(errorLike.parents)) {
        parents.push.apply(parents, __spreadArray([], __read(errorLike.parents), false));
    }
    errorLike.parents = [];
    // error now has all the correct types
    var errorDetailed = errorLike;
    try {
        // ensure parents are valid, and not duplicated
        for (var parents_1 = __values(parents), parents_1_1 = parents_1.next(); !parents_1_1.done; parents_1_1 = parents_1.next()) {
            var parent = parents_1_1.value;
            assertErrorLike(parent);
            if (errorDetailed.parents.indexOf(parent) === -1) {
                errorDetailed.parents.push(parent);
            }
        }
    }
    catch (e_1_1) { e_1 = { error: e_1_1 }; }
    finally {
        try {
            if (parents_1_1 && !parents_1_1.done && (_a = parents_1.return)) _a.call(parents_1);
        }
        finally { if (e_1) throw e_1.error; }
    }
    // adjust the properties if necessary
    if (errorDetailed.code &&
        excludes(errorDetailed.message, "".concat(errorDetailed.code, ": "))) {
        errorDetailed.message = "".concat(errorDetailed.code, ": ").concat(errorDetailed.message);
        errorDetailed.stack = "".concat(errorDetailed.code, ": ").concat(errorDetailed.stack);
    }
    if (errorDetailed.level &&
        excludes(errorDetailed.message, "".concat(errorDetailed.level, ": "))) {
        errorDetailed.message = "".concat(errorDetailed.level, ": ").concat(errorDetailed.message);
        errorDetailed.stack = "".concat(errorDetailed.level, ": ").concat(errorDetailed.stack);
    }
    try {
        for (var _j = __values(errorDetailed.parents), _k = _j.next(); !_k.done; _k = _j.next()) {
            var parent = _k.value;
            if (excludes(errorDetailed.message, "\n\u21AA ".concat(parent.message))) {
                errorDetailed.message = "".concat(errorDetailed.message, "\n\u21AA ").concat(parent.message || parent);
                errorDetailed.stack = "".concat(errorDetailed.stack, "\n\u21AA ").concat(parent.stack || parent.message || parent.parent);
            }
        }
    }
    catch (e_2_1) { e_2 = { error: e_2_1 }; }
    finally {
        try {
            if (_k && !_k.done && (_b = _j.return)) _b.call(_j);
        }
        finally { if (e_2) throw e_2.error; }
    }
    // return
    return errorDetailed;
}
