import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from ncc.utils.code_util.apex.apex_code_utility import ApexCodeUtil

apex_code_util = ApexCodeUtil()

sample_code = """
    public class AccountWithContacts {
    // Method to fetch accounts and their related contacts
        public static void getAccountsWithContacts() {
            // Query to fetch accounts with related contacts
            List<Account> accounts = [SELECT Id, Name, (SELECT Id, LastName, Email FROM Contacts) FROM Account LIMIT 10];

            // Iterate through accounts and print account and contact information to the debug log
            for (Account acc : accounts) {
                System.debug('Account Name: ' + acc.Name);

                for (Contact con : acc.Contacts) {
                    System.debug('Contact Name: ' + con.LastName + ', Email: ' + con.Email);
                }
            }
        }
    }
    """

new_code_snippet = apex_code_util.rename_identifiers(sample_code)
print(new_code_snippet)

