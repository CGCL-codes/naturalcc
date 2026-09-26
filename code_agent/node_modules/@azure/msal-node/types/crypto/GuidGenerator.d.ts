import { IGuidGenerator } from "@azure/msal-common/node";
export declare class GuidGenerator implements IGuidGenerator {
    /**
     * Generates a random [RFC 4122](https://www.rfc-editor.org/rfc/rfc4122.txt) version 4 UUID. The UUID is generated using a
     * cryptographic pseudorandom number generator.
     */
    generateGuid(): string;
    /**
     * verifies if a string is  GUID
     * @param guid
     */
    isGuid(guid: string): boolean;
}
//# sourceMappingURL=GuidGenerator.d.ts.map