"use strict";
/* eslint-disable n/no-sync -- we want the sync methods, as it is a sync operation */
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.loadEdition = loadEdition;
exports.assertEdition = assertEdition;
exports.isValidEdition = isValidEdition;
exports.assertEditions = assertEditions;
exports.isCompatibleVersion = isCompatibleVersion;
exports.isCompatibleEngines = isCompatibleEngines;
exports.isCompatibleEdition = isCompatibleEdition;
exports.determineEdition = determineEdition;
exports.solicitEdition = solicitEdition;
exports.requirePackage = requirePackage;
var version_range_1 = __importDefault(require("version-range"));
var path_1 = require("path");
var process_1 = require("process");
var fs_1 = require("fs");
var util_js_1 = require("./util.js");
/**
 * Load the {@link Edition} with the loader.
 * @param edition - The edition to load
 * @param opts - The loading options containing the loader function and optional path configurations
 * @returns The result of the loaded edition.
 * @throws If failed to load, an error is thrown with the reason.
 */
function loadEdition(edition, opts) {
    var entry = (0, path_1.resolve)(opts.cwd || '', edition.directory, opts.entry || edition.entry || '');
    if (opts.loader == null) {
        throw (0, util_js_1.detailedError)({
            message: "Could not load the edition [".concat(edition.description, "] as no loader was specified. This is probably due to a testing misconfiguration."),
            code: 'editions-autoloader-loader-missing',
            level: 'fatal',
        });
    }
    try {
        return opts.loader.call(edition, entry);
    }
    catch (loadError) {
        // Note the error with more details
        throw (0, util_js_1.detailedError)({
            message: "Failed to load the entry [".concat(entry, "] of edition [").concat(edition.description, "]."),
            code: 'editions-autoloader-loader-failed',
            level: 'fatal',
        }, loadError);
    }
}
/**
 * Asserts that the the {@link Edition} has all the required properties.
 * @param edition - The {@link Edition} to validate
 * @throws if invalid
 */
function assertEdition(edition) {
    if (!edition ||
        typeof edition !== 'object' ||
        !('description' in edition) ||
        !edition.description ||
        !('directory' in edition) ||
        !edition.directory ||
        !('entry' in edition) ||
        !edition.entry ||
        !('engines' in edition) ||
        edition.engines == null) {
        throw (0, util_js_1.detailedError)({
            message: "An edition must have its [description, directory, entry, engines] fields defined, yet all this edition defined were [".concat(Object.keys(edition).join(', '), "]"),
            code: 'editions-autoloader-invalid-edition',
            level: 'fatal',
        });
    }
}
/**
 * Verify the {@link Edition} has all the required properties.
 * @deprecated Use {@link isValidEdition} instead.
 * @param edition - The edition to validate
 * @returns if valid
 * @throws if invalid
 */
function isValidEdition(edition) {
    assertEdition(edition);
    return true;
}
/**
 * Asserts that the provided {@link Editions} array is valid and non-empty.
 * @param editions - The {@link Editions} array to validate
 * @throws if editions is missing, not an array, or empty
 */
function assertEditions(editions) {
    if (!editions || !Array.isArray(editions) || editions.length === 0) {
        throw (0, util_js_1.detailedError)({
            message: 'No editions were specified.',
            code: 'editions-autoloader-editions-missing',
        });
    }
}
/**
 * Is this {@link Edition} suitable for these versions?
 * @param range - The version range to check compatibility against
 * @param version - The actual version to check
 * @param opts - The range options for configuring how ranges are handled
 * @returns if compatible
 * @throws if incompatible
 */
function isCompatibleVersion(range, version, opts) {
    // prepare
    var broadenRange = opts.broadenRange;
    if (!version)
        throw (0, util_js_1.detailedError)({
            message: "No version was specified to compare the range [".concat(range, "] against"),
            code: 'editions-autoloader-engine-version-missing',
            level: 'fatal',
        });
    if (range == null || range === '')
        throw (0, util_js_1.detailedError)({
            message: "The edition range was not specified, so unable to compare against the version [".concat(version, "]"),
            code: 'editions-autoloader-engine-range-missing',
        });
    if (range === false)
        throw (0, util_js_1.detailedError)({
            message: "The edition range does not support this engine",
            code: 'editions-autoloader-engine-unsupported',
        });
    if (range === true)
        return true;
    // original range
    try {
        if ((0, version_range_1.default)(version, range))
            return true;
    }
    catch (error) {
        throw (0, util_js_1.detailedError)({
            message: "The range [".concat(range, "] was invalid, something is wrong with the Editions definition."),
            code: 'editions-autoloader-invalid-range',
            level: 'fatal',
        }, error);
    }
    // broadened range
    // https://github.com/bevry/editions/blob/master/HISTORY.md#v210-2018-november-15
    // If none of the editions for a package match the current node version, editions will try to find a compatible package by converting strict version ranges likes 4 || 6 || 8 || 10 to looser ones like >=4, and if that fails, then it will attempt to load the last edition for the environment.
    // This brings editions handling of engines closer in line with how node handles it, which is as a warning/recommendation, rather than a requirement/enforcement.
    // This has the benefit that edition authors can specify ranges as the specific versions that they have tested the edition against that pass, rather than having to omit that information for runtime compatibility.
    // As such editions will now automatically select the edition with guaranteed support for the environment, and if there are none with guaranteed support, then editions will select the one is most likely supported, and if there are none that are likely supported, then it will try the last edition, which should be the most compatible edition.
    // This is timely, as node v11 is now the version most developers use, yet if edition authors specified only LTS releases, then the editions autoloader would reject loading on v11, despite compatibility being likely with the most upper edition.
    // NOTE: That there is only one broadening chance per package, once a broadened edition has been returned, a load will be attempted, and if it fails, then the package failed. This is intentional.
    if (broadenRange === true) {
        // check if range can be broadened, validate it and extract
        var broadenedRangeRegex = /^\s*([0-9.]+)\s*(\|\|\s*[0-9.]+\s*)*$/;
        var broadenedRangeMatch = broadenedRangeRegex.exec(range);
        var lowestVersion = (broadenedRangeMatch && broadenedRangeMatch[1]) || '';
        // ^ can't do number conversion, as 1.1.1 is not a number
        // this also converts 0 to '' which is what we want for the next check
        // confirm the validation
        if (lowestVersion === '')
            throw (0, util_js_1.detailedError)({
                message: "The range [".concat(range, "] is not able to be broadened, only ranges in format of [lowest] or [lowest || ... || ... ] can be broadened. Update the Editions definition and try again."),
                code: 'editions-autoloader-unsupported-broadened-range',
                level: 'fatal',
            });
        // create the broadened range, and attempt that
        var broadenedRange = ">= ".concat(lowestVersion);
        try {
            if ((0, version_range_1.default)(version, broadenedRange))
                return true;
        }
        catch (error) {
            throw (0, util_js_1.detailedError)({
                message: "The broadened range [".concat(broadenedRange, "] was invalid, something is wrong within Editions."),
                code: 'editions-autoloader-invalid-broadened-range',
                level: 'fatal',
            }, error);
        }
        // broadened range was incompatible
        throw (0, util_js_1.detailedError)({
            message: "The edition range [".concat(range, "] does not support this engine version [").concat(version, "], even when broadened to [").concat(broadenedRange, "]"),
            code: 'editions-autoloader-engine-incompatible-broadened-range',
        });
    }
    // give up
    throw (0, util_js_1.detailedError)({
        message: "The edition range [".concat(range, "] does not support this engine version [").concat(version, "]"),
        code: 'editions-autoloader-engine-incompatible-original',
    });
}
/**
 * Checks that the provided engines are compatible against the provided versions.
 * @param engines - The engine specifications to check compatibility for
 * @param opts - The version options containing environment versions and range settings
 * @returns if compatible
 * @throws if incompatible
 */
function isCompatibleEngines(engines, opts) {
    // PRepare
    var versions = opts.versions;
    // Check engines exist
    if (!engines) {
        throw (0, util_js_1.detailedError)({
            message: "The edition had no engines to compare against the environment",
            code: 'editions-autoloader-invalid-engines',
        });
    }
    // Check versions exist
    if (!versions) {
        throw (0, util_js_1.detailedError)({
            message: "No versions were supplied to compare the engines against",
            code: 'editions-autoloader-invalid-versions',
            level: 'fatal',
        });
    }
    // Check each version
    var compatible = false;
    for (var key in engines) {
        if (Object.prototype.hasOwnProperty.call(engines, key)) {
            // deno's std/node/process provides both `deno` and `node` keys
            // so we don't won't to compare node when it is actually deno
            if (key === 'node' && versions.deno)
                continue;
            // prepare
            var engine = engines[key];
            var version = versions[key];
            // skip for engines this edition does not care about
            if (version == null)
                continue;
            // check compatibility against all the provided engines it does care about
            try {
                isCompatibleVersion(engine, version, opts);
                compatible = true;
                // if any incompatibility, it is thrown, so no need to set to false
            }
            catch (rangeError) {
                throw (0, util_js_1.detailedError)({
                    message: "The engine [".concat(key, "] range of [").concat(engine, "] was not compatible against version [").concat(version, "]."),
                    code: 'editions-autoloader-engine-error',
                }, rangeError);
            }
        }
    }
    // if there were no matching engines, then throw
    if (!compatible) {
        throw (0, util_js_1.detailedError)({
            message: "There were no supported engines in which this environment provides.",
            code: 'editions-autoloader-engine-mismatch',
        });
    }
    // valid
    return true;
}
/**
 * Checks that the {@link Edition} is compatible against the provided versions.
 * @param edition - The edition to check for compatibility
 * @param opts - The version options containing environment versions and range settings
 * @returns if compatible
 * @throws if incompatible
 */
function isCompatibleEdition(edition, opts) {
    try {
        return isCompatibleEngines(edition.engines, opts);
    }
    catch (compatibleError) {
        throw (0, util_js_1.detailedError)({
            message: "The edition [".concat(edition.description, "] is not compatible with this environment."),
            code: 'editions-autoloader-edition-incompatible',
        }, compatibleError);
    }
}
/**
 * Determine which edition should be loaded.
 * If {@link VersionOptions.broadenRange} is unspecified (the default behavior), then we attempt to determine a suitable edition without broadening the range, and if that fails, then we try again with the range broadened.
 * @param editions - The array of editions to choose from
 * @param opts - The version options containing environment versions and range settings
 * @returns any suitable editions
 * @throws if no suitable editions
 */
function determineEdition(editions, opts) {
    // Prepare
    assertEditions(editions);
    var broadenRange = opts.broadenRange;
    // Cycle through the editions determining the above
    var failure = null;
    for (var i = 0; i < editions.length; ++i) {
        var edition = editions[i];
        try {
            assertEdition(edition);
            isCompatibleEdition(edition, opts);
            // Success! Return the edition
            return edition;
        }
        catch (error) {
            // Failure!
            if ((error === null || error === void 0 ? void 0 : error.level) === 'fatal') {
                throw (0, util_js_1.detailedError)({
                    message: "Unable to determine a suitable edition due to failure.",
                    code: 'editions-autoloader-fatal',
                    level: 'fatal',
                }, error);
            }
            else if (failure) {
                failure = (0, util_js_1.detailedError)(error, failure);
            }
            else {
                failure = error;
            }
        }
    }
    // Report the failure from above
    if (failure) {
        // try broadened
        if (broadenRange == null)
            try {
                // return if broadening successfully returned an edition
                var broadenedEdition = determineEdition(editions, __assign(__assign({}, opts), { broadenRange: true }));
                return __assign(__assign({}, broadenedEdition), { 
                    // bubble the circumstances up in case the loading of the broadened edition fails and needs to be reported
                    debugging: (0, util_js_1.detailedError)({
                        message: "The edition ".concat(broadenedEdition.description, " was selected to be force loaded as its range was broadened."),
                        code: 'editions-autoloader-attempt-broadened',
                    }) });
            }
            catch (error) {
                throw (0, util_js_1.detailedError)({
                    message: "Unable to determine a suitable edition, even after broadening.",
                    code: 'editions-autoloader-none-broadened',
                }, error);
            }
        // fail
        throw (0, util_js_1.detailedError)({
            message: "Unable to determine a suitable edition, as none were suitable.",
            code: 'editions-autoloader-none-suitable',
        }, failure);
    }
    // this should never reach here
    throw (0, util_js_1.detailedError)({
        message: "Unable to determine a suitable edition, as an unexpected pathway occurred.",
        code: 'editions-autoloader-never',
    });
}
/**
 * Determine which edition should be loaded, and attempt to load it.
 * @param editions - The array of editions to choose from
 * @param opts - The solicit options containing loader, versions, and path configurations
 * @returns the loaded result of the suitable edition
 * @throws if no suitable editions, or the edition failed to load
 */
function solicitEdition(editions, opts) {
    var edition = determineEdition(editions, opts);
    try {
        // Load the edition
        return loadEdition(edition, opts);
    }
    catch (error) {
        // Failure!
        if (edition.debugging) {
            throw (0, util_js_1.detailedError)(error, edition.debugging);
        }
        else {
            throw error;
        }
    }
}
/**
 * Cycle through the editions for a package, determine the compatible edition, and load it.
 * @param cwd - The current working directory path where the package.json is located
 * @param loader - The loader function to use for loading the selected edition
 * @param entry - The entry point to load from the selected edition
 * @returns the loaded result of the suitable edition
 * @throws if no suitable editions, or if the edition failed to load
 */
function requirePackage(cwd, loader, entry) {
    var packagePath = (0, path_1.resolve)(cwd || '', 'package.json');
    try {
        // Load editions
        var editions = JSON.parse((0, fs_1.readFileSync)(packagePath, 'utf8')).editions;
        assertEditions(editions);
        // Load edition
        return solicitEdition(editions, {
            versions: process_1.versions,
            cwd: cwd,
            loader: loader,
            entry: entry,
        });
    }
    catch (error) {
        // Failure!
        throw (0, util_js_1.detailedError)({
            message: "Unable to determine a suitable edition for the package [".concat(packagePath, "] and entry [").concat(entry, "]"),
            code: 'editions-autoloader-package',
        }, error);
    }
}
