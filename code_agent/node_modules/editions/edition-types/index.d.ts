import { ErrorDetailed } from './util.js';
/** A version range specification that can be a string (semver range) or boolean (true for any version, false for none) */
export type Range = string | boolean;
/** Engine specifications that can be false (unsupported) or a record of engine names to their version ranges */
export type Engines = false | Record<string, Range>;
/** A record of environment versions, mapping environment names to their version strings */
export type Versions = Record<string, string>;
/**
 * Edition entries must conform to the following specification.
 * @example
 * ``` json
 * {
 *   "description": "esnext source code with require for modules",
 *   "directory": "source",
 *   "entry": "index.js",
 *   "tags": [
 *     "javascript",
 *     "esnext",
 *     "require"
 *   ],
 *   "engines": {
 *     "node": ">=6",
 *     "browsers": "defaults"
 *   }
 * }
 * ```
 */
export interface Edition {
    /**
     * Use this property to describe the edition in human readable terms. Such as what it does and who it is for. It is used to reference the edition in user facing reporting, such messages.
     * @example
     * ``` json
     * "esnext source code with require for modules"
     * ```
     */
    description: string;
    /**
     * The location to where this directory is located. It should be a relative path from the `package.json` file.
     * @example
     * ``` json
     * "source"
     * ```
     */
    directory: string;
    /**
     * The default entry location for this edition, relative to the edition's directory.
     * @example
     * ``` json
     * "index.js"
     * ```
     */
    entry: string;
    /**
     * Any keywords you wish to associate to the edition. Useful for various ecosystem tooling, such as automatic ESNext lint configuration if the `esnext` tag is present in the source edition tags.
     * @example
     * ``` json
     * ["javascript", "esnext", "require"]
     * ```
     */
    tags?: string[];
    /**
     * This field is used to specific which environments this edition supports.
     * If `false` this edition does not any environment.
     * If `deno` is a string, it should be a semver range of Deno versions that the edition targets.
     * If `node` is a string, it should be a semver range of Node.js versions that the edition targets.
     * If `browsers` is a string, it should be a [browserlist](https://github.com/browserslist/browserslist) value of the specific browser values the edition targets. If multiple engines are truthy, it indicates that this edition is compatible with those multiple environments.
     * @example
     * ``` json
     * {
     *   "deno": ">=1",
     *   "node": ">=6",
     *   "browsers": "defaults"
     * }
     * ```
     */
    engines: Engines;
    /** If this edition fails to load, then this property provides any accompanying information. */
    debugging?: ErrorDetailed;
}
/** Editions should be ordered from most preferable first to least desirable last. The source edition should always be first, proceeded by compiled editions. */
export type Editions = Edition[];
/** Options for specifying paths when loading editions */
export interface PathOptions {
    /** If provided, this edition entry is used instead of the default entry. */
    entry: string;
    /** If provided, edition loading will be resolved against this. */
    cwd: string;
}
/**
 * The method that will load the entry of the edition.
 * For CJS files this should be set to the `require` method.
 * For MJS files this should be set to `(path: string) => import(path)`.
 */
export type Loader = (this: Edition, path: string) => unknown;
/** Options for loading editions, including optional path configuration and a required loader function */
export interface LoaderOptions extends Partial<PathOptions> {
    loader: Loader;
}
/** Options for configuring how version ranges are handled */
export interface RangeOptions {
    /** If `true`, then ranges such as `x || y` are changed to `>=x`. */
    broadenRange?: boolean;
}
/** Options that combine range configuration with version information */
export interface VersionOptions extends RangeOptions {
    /** The versions of our current environment. */
    versions: Versions;
}
/** Combined options for soliciting (determining and loading) editions */
export interface SolicitOptions extends LoaderOptions, VersionOptions {
}
/**
 * Load the {@link Edition} with the loader.
 * @param edition - The edition to load
 * @param opts - The loading options containing the loader function and optional path configurations
 * @returns The result of the loaded edition.
 * @throws If failed to load, an error is thrown with the reason.
 */
export declare function loadEdition<T>(edition: Edition, opts: LoaderOptions): T;
/**
 * Asserts that the the {@link Edition} has all the required properties.
 * @param edition - The {@link Edition} to validate
 * @throws if invalid
 */
export declare function assertEdition(edition: unknown): asserts edition is Edition;
/**
 * Verify the {@link Edition} has all the required properties.
 * @deprecated Use {@link isValidEdition} instead.
 * @param edition - The edition to validate
 * @returns if valid
 * @throws if invalid
 */
export declare function isValidEdition(edition: Edition): true;
/**
 * Asserts that the provided {@link Editions} array is valid and non-empty.
 * @param editions - The {@link Editions} array to validate
 * @throws if editions is missing, not an array, or empty
 */
export declare function assertEditions(editions: unknown): asserts editions is Editions;
/**
 * Is this {@link Edition} suitable for these versions?
 * @param range - The version range to check compatibility against
 * @param version - The actual version to check
 * @param opts - The range options for configuring how ranges are handled
 * @returns if compatible
 * @throws if incompatible
 */
export declare function isCompatibleVersion(range: Range, version: string, opts: RangeOptions): true;
/**
 * Checks that the provided engines are compatible against the provided versions.
 * @param engines - The engine specifications to check compatibility for
 * @param opts - The version options containing environment versions and range settings
 * @returns if compatible
 * @throws if incompatible
 */
export declare function isCompatibleEngines(engines: Engines, opts: VersionOptions): true;
/**
 * Checks that the {@link Edition} is compatible against the provided versions.
 * @param edition - The edition to check for compatibility
 * @param opts - The version options containing environment versions and range settings
 * @returns if compatible
 * @throws if incompatible
 */
export declare function isCompatibleEdition(edition: Edition, opts: VersionOptions): true;
/**
 * Determine which edition should be loaded.
 * If {@link VersionOptions.broadenRange} is unspecified (the default behavior), then we attempt to determine a suitable edition without broadening the range, and if that fails, then we try again with the range broadened.
 * @param editions - The array of editions to choose from
 * @param opts - The version options containing environment versions and range settings
 * @returns any suitable editions
 * @throws if no suitable editions
 */
export declare function determineEdition(editions: Editions, opts: VersionOptions): Edition;
/**
 * Determine which edition should be loaded, and attempt to load it.
 * @param editions - The array of editions to choose from
 * @param opts - The solicit options containing loader, versions, and path configurations
 * @returns the loaded result of the suitable edition
 * @throws if no suitable editions, or the edition failed to load
 */
export declare function solicitEdition<T>(editions: Editions, opts: SolicitOptions): T;
/**
 * Cycle through the editions for a package, determine the compatible edition, and load it.
 * @param cwd - The current working directory path where the package.json is located
 * @param loader - The loader function to use for loading the selected edition
 * @param entry - The entry point to load from the selected edition
 * @returns the loaded result of the suitable edition
 * @throws if no suitable editions, or if the edition failed to load
 */
export declare function requirePackage<T>(cwd: PathOptions['cwd'], loader: LoaderOptions['loader'], entry: PathOptions['entry']): T;
//# sourceMappingURL=index.d.ts.map