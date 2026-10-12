"""
WHAT: Bank ledger and entry models enforcing balance invariants, limit caps, and per-bank privacy.
WHY: Guarantees double-entry recording, prevents negative balances, and models per-bank visibility.
TERM: ledger: The authoritative record of financial transactions and balances for accounts within one bank.
TERM: ledger entry: A single debit or credit line item permanently written to the ledger.
"""

from dataclasses import dataclass, field


class AccountNotInThisBank(Exception):
    """
    WHAT: Raised when a ledger is queried about an account managed by another bank.
    WHY: Enforces per-bank visibility boundaries so banks cannot observe external accounts directly.
    """


@dataclass
class LedgerEntry:
    """
    WHAT: Represents a single immutable financial movement on an account.
    WHY: Provides the audit trail needed to recompute and verify balances.
    """

    entry_id: str
    minute: int
    account_id: str
    amount_paise: int
    kind: str
    transfer_id: str


@dataclass
class Ledger:
    """
    WHAT: Maintains accounts, balances in integer paise, and entries for a single bank.
    WHY: Enforces that balances never drop below zero and transfers respect the UPI cap.
    """

    bank_index: int
    upi_limit_paise: int
    accounts: set[str] = field(default_factory=set)
    balances: dict[str, int] = field(default_factory=dict)
    entries: list[LedgerEntry] = field(default_factory=list)
    _entry_sequence: int = 0

    def add_account(self, account_id: str, opening_balance_paise: int = 0, minute: int = 0) -> None:
        """
        WHAT: Registers a new account with an initial opening balance.
        WHY: Establishes account presence in this bank and logs the initial funds.
        """
        if opening_balance_paise < 0:
            raise ValueError(f"Opening balance cannot be negative, got {opening_balance_paise}")

        self.accounts.add(account_id)
        self.balances[account_id] = opening_balance_paise
        if opening_balance_paise > 0:
            self._entry_sequence += 1
            entry = LedgerEntry(
                entry_id=f"ent_{self.bank_index}_{self._entry_sequence}",
                minute=minute,
                account_id=account_id,
                amount_paise=opening_balance_paise,
                kind="opening",
                transfer_id=f"open_{account_id}",
            )
            self.entries.append(entry)

    def post_debit(self, account_id: str, amount_paise: int, minute: int, transfer_id: str) -> LedgerEntry:
        """
        WHAT: Deducts amount_paise from an account and appends a transfer_debit entry.
        WHY: Executes the outgoing leg of a transfer after validating limits and funds.
        """
        if account_id not in self.accounts:
            raise AccountNotInThisBank(f"Account '{account_id}' is not in bank {self.bank_index}")
        if amount_paise <= 0:
            raise ValueError(f"Debit amount must be positive, got {amount_paise}")
        if amount_paise > self.upi_limit_paise:
            raise ValueError(f"Debit leg {amount_paise} exceeds UPI limit {self.upi_limit_paise}")
        if self.balances[account_id] < amount_paise:
            raise ValueError(f"Insufficient funds: balance {self.balances[account_id]} < {amount_paise}")

        self.balances[account_id] -= amount_paise
        self._entry_sequence += 1
        entry = LedgerEntry(
            entry_id=f"ent_{self.bank_index}_{self._entry_sequence}",
            minute=minute,
            account_id=account_id,
            amount_paise=-amount_paise,
            kind="transfer_debit",
            transfer_id=transfer_id,
        )
        self.entries.append(entry)
        return entry

    def post_credit(self, account_id: str, amount_paise: int, minute: int, transfer_id: str) -> LedgerEntry:
        """
        WHAT: Adds amount_paise to an account and appends a transfer_credit entry.
        WHY: Executes the incoming leg of a transfer after validating limits.
        """
        if account_id not in self.accounts:
            raise AccountNotInThisBank(f"Account '{account_id}' is not in bank {self.bank_index}")
        if amount_paise <= 0:
            raise ValueError(f"Credit amount must be positive, got {amount_paise}")
        if amount_paise > self.upi_limit_paise:
            raise ValueError(f"Credit leg {amount_paise} exceeds UPI limit {self.upi_limit_paise}")

        self.balances[account_id] += amount_paise
        self._entry_sequence += 1
        entry = LedgerEntry(
            entry_id=f"ent_{self.bank_index}_{self._entry_sequence}",
            minute=minute,
            account_id=account_id,
            amount_paise=amount_paise,
            kind="transfer_credit",
            transfer_id=transfer_id,
        )
        self.entries.append(entry)
        return entry

    def balance_of(self, account_id: str) -> int:
        """
        WHAT: Returns the current stored balance in integer paise.
        WHY: Provides constant-time balance lookup while checking bank ownership.
        """
        if account_id not in self.accounts:
            raise AccountNotInThisBank(f"Account '{account_id}' is not in bank {self.bank_index}")
        return self.balances[account_id]

    def recompute_balance(self, account_id: str) -> int:
        """
        WHAT: Calculates the account balance by summing all of its historical ledger entries.
        WHY: Validates that the fast stored balance exactly matches the immutable audit log.
        """
        if account_id not in self.accounts:
            raise AccountNotInThisBank(f"Account '{account_id}' is not in bank {self.bank_index}")
        return sum(e.amount_paise for e in self.entries if e.account_id == account_id)
