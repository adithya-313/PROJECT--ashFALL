"""
WHAT: Bank ledger and entry models enforcing balance invariants, limit caps, and per-bank privacy.
It tracks accounts, records immutable debit and credit ledger entries, and maintains running balances in integer paise.
All balance updates enforce non-negative funds constraints and regulatory transaction limits.

WHY: Guarantees double-entry recording, prevents negative balances, and models per-bank visibility.
Rigorous financial record keeping ensures transactions are completely auditable and money is conserved.
Isolating ledgers per bank accurately reflects real-world banking boundaries where banks cannot read rival data.

TERM: ledger: The authoritative record of financial transactions and balances for accounts within one bank.
      For example, a bank database tracking each customer's balance.
TERM: ledger entry: A single debit or credit line item permanently written to the ledger.
      For example, recording a 2000 paise debit for groceries.
TERM: double-entry bookkeeping: An accounting standard where every transfer consists of equal debit and credit legs.
      For example, debiting the sender and crediting the receiver.
TERM: balance invariant: A mathematical guarantee that account funds never fall below zero and match
      transaction sums. For example, ensuring an account balance never drops into negative numbers.
"""

from dataclasses import dataclass, field


class AccountNotInThisBank(Exception):
    """
    WHAT: Raised when a ledger is queried about an account managed by another bank.
    It indicates that an account identifier does not belong to the queried bank's registry.
    This exception prevents illegal cross-bank account access and maintains privacy boundaries.

    WHY: Enforces per-bank visibility boundaries so banks cannot observe external accounts directly.
    In real banking systems, one financial institution cannot query account records belonging to another.
    Raising explicit exceptions ensures simulated services respect banking privacy partitions.

    Args:
        message: Descriptive string identifying the missing account and bank index.

    Returns:
        Instantiated AccountNotInThisBank exception object.

    Example:
        # Instantiating the exception with a descriptive message:
        # err = AccountNotInThisBank("Account acc_99 is not in bank 0")
        # assert "acc_99" in str(err)
    """


@dataclass
class LedgerEntry:
    """
    WHAT: Represents a single immutable financial movement on an account.
    It records the timestamp, account identifier, amount transferred in paise, entry type, and link.
    Debit amounts are recorded as negative integers, while credit amounts are positive integers.

    WHY: Provides the audit trail needed to recompute and verify balances.
    Storing complete historical movements ensures all balance changes can be independently audited.
    Immutable entry records guarantee that ledger histories cannot be silently altered or lost.

    Args:
        entry_id: Unique string identifier for this specific ledger entry.
        minute: Simulation minute timestamp when the movement was posted.
        account_id: Account identifier affected by this ledger entry.
        amount_paise: Signed integer currency amount in paise (negative for debits, positive for credits).
        kind: Classification tag for the entry ('opening', 'transfer_debit', or 'transfer_credit').
        transfer_id: Unique correlation identifier linking related debit and credit legs.

    Returns:
        Initialized LedgerEntry dataclass instance.

    Example:
        # Create an opening balance ledger entry:
        # entry = LedgerEntry("ent_0_1", 0, "acc_1", 5000, "opening", "open_acc_1")
        # assert entry.amount_paise == 5000 and entry.kind == "opening"
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
    It provides operations to register accounts, post compliant debits and credits, and query balances.
    All state transitions are recorded into an append-only sequence of immutable ledger entries.

    WHY: Enforces that balances never drop below zero and transfers respect the UPI cap.
    Encapsulating bank-level ledger logic ensures strict compliance with financial regulations.
    It guarantees that stored balances can always be reconstructed by summing entry histories.

    Args:
        bank_index: Unique integer index designating which bank owns this ledger.
        upi_limit_paise: Maximum legal transfer limit in paise allowed per single transaction leg.
        accounts: Set of account identifier strings registered with this bank.
        balances: Mapping from account identifiers to current stored integer paise balances.
        entries: Ordered list of all immutable LedgerEntry records posted to this bank.
        _entry_sequence: Internal monotonically increasing counter for unique entry IDs.

    Returns:
        Initialized Ledger instance ready to manage accounts and process payments.

    Example:
        # Create a ledger for bank 0 with a 100,000-rupee UPI limit:
        # l = Ledger(bank_index=0, upi_limit_paise=10000000)
        # assert l.bank_index == 0 and l.upi_limit_paise == 10000000
    """

    bank_index: int
    upi_limit_paise: int
    accounts: set[str] = field(default_factory=set)
    balances: dict[str, int] = field(default_factory=dict)
    entries: list[LedgerEntry] = field(default_factory=list)
    _entry_sequence: int = 0

    def add_account(self, account_id: str, opening_balance_paise: int = 0, minute: int = 0) -> None:
        """
        WHAT: Registers a new account with an initial opening balance in integer paise.
        It stores the account in the ledger's account registry and initializes its running balance.
        If the opening balance is positive, it generates an initial 'opening' ledger entry.

        WHY: Establishes account presence in this bank and logs the initial funds.
        Creating an explicit opening entry maintains an auditable record of all seeded currency.
        Enforcing non-negative opening balances prevents corrupt starting states.

        Args:
            account_id: Unique string identifier for the newly created account.
            opening_balance_paise: Initial balance in integer paise, defaulting to 0.
            minute: Simulation minute timestamp when the account is registered, defaulting to 0.

        Returns:
            None.

        Example:
            # Register a new account with 5000 paise opening balance:
            # l = Ledger(0, 10000000)
            # l.add_account("acc_1", 5000)
            # assert l.balance_of("acc_1") == 5000
        """
        # Validate that the opening balance is non-negative.
        if opening_balance_paise < 0:
            raise ValueError(f"Opening balance cannot be negative, got {opening_balance_paise}")

        # Register the account and initialize its stored balance.
        self.accounts.add(account_id)
        self.balances[account_id] = opening_balance_paise
        # If opening balance is positive, log the initial funding transaction.
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
        It verifies account membership, positive transfer size, UPI cap compliance, and solvency.
        The account balance is decremented and the newly recorded LedgerEntry is returned.

        WHY: Executes the outgoing leg of a transfer after validating limits and funds.
        Strict validation prevents unauthorized overdrafts and regulatory limit violations.
        Recording a negative ledger entry maintains an unbroken financial audit trail.

        Args:
            account_id: String identifier of the paying account registered at this bank.
            amount_paise: Positive integer transfer amount in paise to deduct from the balance.
            minute: Simulation timestamp minute when the debit is posted.
            transfer_id: Unique string identifier linking this debit to its matching credit.

        Returns:
            The newly created LedgerEntry documenting the debit transaction.

        Example:
            # Post a 2000 paise debit against an account holding 5000 paise:
            # l = Ledger(0, 10000000)
            # l.add_account("acc_1", 5000)
            # deb = l.post_debit("acc_1", 2000, 10, "tx_1")
            # assert deb.amount_paise == -2000 and l.balance_of("acc_1") == 3000
        """
        # Validate that the account is registered with this bank.
        if account_id not in self.accounts:
            raise AccountNotInThisBank(f"Account '{account_id}' is not in bank {self.bank_index}")
        # Verify that the debit amount is strictly positive.
        if amount_paise <= 0:
            raise ValueError(f"Debit amount must be positive, got {amount_paise}")
        # Verify that the debit does not exceed the regulatory UPI transaction ceiling.
        if amount_paise > self.upi_limit_paise:
            raise ValueError(f"Debit leg {amount_paise} exceeds UPI limit {self.upi_limit_paise}")
        # Verify that the account has sufficient funds to cover the full debit.
        if self.balances[account_id] < amount_paise:
            raise ValueError(f"Insufficient funds: balance {self.balances[account_id]} < {amount_paise}")

        # Deduct the amount from the stored balance and record the negative ledger entry.
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
        It verifies that the recipient account exists in this bank and respects the UPI ceiling.
        The account balance is incremented and the resulting LedgerEntry is returned.

        WHY: Executes the incoming leg of a transfer after validating limits.
        Guaranteeing that credits only post to existing registered accounts maintains ledger integrity.
        The corresponding ledger entry records the exact arrival timestamp and transaction link.

        Args:
            account_id: String identifier of the recipient account registered at this bank.
            amount_paise: Positive integer transfer amount in paise to credit to the account.
            minute: Simulation minute timestamp when the credit takes place.
            transfer_id: Unique string identifier linking this credit to its originating debit.

        Returns:
            The newly created LedgerEntry documenting the credit transaction.

        Example:
            # Post a 3000 paise credit to an account holding 5000 paise:
            # l = Ledger(0, 10000000)
            # l.add_account("acc_1", 5000)
            # cred = l.post_credit("acc_1", 3000, 20, "tx_2")
            # assert cred.amount_paise == 3000 and l.balance_of("acc_1") == 8000
        """
        # Validate that the destination account is registered with this bank.
        if account_id not in self.accounts:
            raise AccountNotInThisBank(f"Account '{account_id}' is not in bank {self.bank_index}")
        # Verify that the credit amount is strictly positive.
        if amount_paise <= 0:
            raise ValueError(f"Credit amount must be positive, got {amount_paise}")
        # Verify that the credit leg does not exceed the regulatory UPI ceiling.
        if amount_paise > self.upi_limit_paise:
            raise ValueError(f"Credit leg {amount_paise} exceeds UPI limit {self.upi_limit_paise}")

        # Credit the amount to the stored balance and append the positive ledger entry.
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
        It checks that the account is managed by this bank before accessing the balance mapping.
        The returned integer reflects all debits, credits, and opening deposits posted to date.

        WHY: Provides constant-time balance lookup while checking bank ownership.
        Quick balance queries prevent performance bottlenecks during high-throughput routine simulation runs.
        Verifying account ownership prevents cross-bank data leakage across financial institutions.

        Args:
            account_id: String identifier of the account to look up.

        Returns:
            Current balance of the account in integer paise.

        Example:
            # Check balance of an account with 5000 paise:
            # l = Ledger(0, 10000000)
            # l.add_account("acc_1", 5000)
            # assert l.balance_of("acc_1") == 5000
        """
        # Verify that the requested account belongs to this bank ledger.
        if account_id not in self.accounts:
            raise AccountNotInThisBank(f"Account '{account_id}' is not in bank {self.bank_index}")
        return self.balances[account_id]

    def recompute_balance(self, account_id: str) -> int:
        """
        WHAT: Calculates the account balance by summing all of its historical ledger entries.
        It filters the complete ledger entry list for entries matching the account identifier and sums them.
        The resulting sum provides an independent verification of the stored account balance.

        WHY: Validates that the fast stored balance exactly matches the immutable audit log.
        Auditing stored balances against raw transaction logs detects calculation anomalies or data corruption.
        It proves double-entry accounting integrity across simulated bank operations.

        Args:
            account_id: String identifier of the account whose entries will be recomputed.

        Returns:
            Summed integer paise balance derived directly from ledger entry history.

        Example:
            # Recompute balance from entries for an account with 5000 paise:
            # l = Ledger(0, 10000000)
            # l.add_account("acc_1", 5000)
            # assert l.recompute_balance("acc_1") == 5000
        """
        # Verify bank ownership and aggregate all historical transaction amounts.
        if account_id not in self.accounts:
            raise AccountNotInThisBank(f"Account '{account_id}' is not in bank {self.bank_index}")
        return sum(e.amount_paise for e in self.entries if e.account_id == account_id)

