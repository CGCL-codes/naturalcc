"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.PackageManager = void 0;
exports.createVSIX = createVSIX;
exports.publish = publish;
exports.listFiles = listFiles;
exports.publishVSIX = publishVSIX;
exports.unpublishVSIX = unpublishVSIX;
const publish_1 = require("./publish");
const package_1 = require("./package");
/**
 * The supported list of package managers.
 * @public
 */
var PackageManager;
(function (PackageManager) {
    PackageManager[PackageManager["Npm"] = 0] = "Npm";
    PackageManager[PackageManager["Yarn"] = 1] = "Yarn";
    PackageManager[PackageManager["None"] = 2] = "None";
})(PackageManager || (exports.PackageManager = PackageManager = {}));
/**
 * Creates a VSIX from the extension in the current working directory.
 * @public
 */
function createVSIX(options = {}) {
    return (0, package_1.packageCommand)(options);
}
/**
 * Publishes the extension in the current working directory.
 * @public
 */
function publish(options = {}) {
    return (0, publish_1.publish)(options);
}
/**
 * Lists the files included in the extension's package.
 * @public
 */
function listFiles(options = {}) {
    return (0, package_1.listFiles)({
        ...options,
        useYarn: options.packageManager === PackageManager.Yarn,
        dependencies: options.packageManager !== PackageManager.None,
    });
}
/**
 * Publishes a pre-build VSIX.
 * @public
 */
function publishVSIX(packagePath, options = {}) {
    return (0, publish_1.publish)({
        packagePath: typeof packagePath === 'string' ? [packagePath] : packagePath,
        ...options,
        targets: typeof options.target === 'string' ? [options.target] : undefined,
        ...{ target: undefined },
    });
}
/**
 * Deletes a specific extension from the marketplace.
 * @public
 */
function unpublishVSIX(options = {}) {
    return (0, publish_1.unpublish)({ force: true, ...options });
}
//# sourceMappingURL=api.js.map