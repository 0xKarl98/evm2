//! OSS-1032 probe: the evm2 `State` overlay of each fork is the single accepted-state authority.
//!
//! Foundry's backend shrinks to a [`Registry`]: backing database handles, saved state of inactive
//! forks, the snapshot registry, and persistent accounts. The [`Executor`] keeps the active fork's
//! state between calls, builds an [`Evm`] per call, and moves the state in and out. Cheatcodes are
//! dispatched from an [`Inspector::call`] hook, which reaches the running [`Evm`] mid-transaction.
//!
//! Only public evm2 API is used. Run the measurement with
//! `cargo test --release -p evm2 --test foundry_registry_probe -- --ignored --nocapture`.

use alloy_consensus::{TxLegacy, transaction::Recovered};
use alloy_primitives::{
    Address, B256, Bytes, TxKind,
    map::{AddressMap, AddressSet, B256Map, HashMap},
};
use evm2::{
    BaseEvmTypes, DatabaseError, Evm, EvmFeatures, ExecutionConfig, Inspector, Precompiles, SpecId,
    Version,
    bytecode::Bytecode,
    env::BlockEnvExt,
    ethereum::{TxEnvelope, ethereum_tx_registry},
    evm::{
        AccountInfo, Cache, DbResult, DynDatabase, EmptyDB, PendingState, State, StateCheckpoint,
        StateSnapshot,
    },
    interpreter::{GasTracker, InstrStop, Interpreter, Message, MessageResult, Word, op},
};
use std::{
    cell::RefCell,
    hint::black_box,
    mem,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

const SPEC: SpecId = SpecId::CANCUN;
const CALLER: Address = Address::with_last_byte(0xca);
/// The test contract; its code is a script of calls, like a test function body.
const TEST: Address = Address::with_last_byte(0x7e);
const COUNTER: Address = Address::with_last_byte(0xc0);
/// A counter whose slot is never preloaded, so reading it hits the backing database.
const COLD_COUNTER: Address = Address::with_last_byte(0xc1);
/// A counter that no backing database has, deployed on one fork like a `setUp` deployment.
const DEPLOYED: Address = Address::with_last_byte(0xd0);
/// Stand-in for the cheatcode address.
const CHEATS: Address = Address::with_last_byte(0xcc);
/// Script contracts the test calls, so that with isolation their cheatcodes run inside a child.
const HANDLER: Address = Address::with_last_byte(0x4a);
const NESTED: Address = Address::with_last_byte(0x4b);
const FAILING: Address = Address::with_last_byte(0x4c);

/// `selectFork(arg)`.
const SELECT_FORK: u8 = 1;
/// `snapshotState()`; ids are assigned in order from 0.
const SNAPSHOT: u8 = 2;
/// `revertToState(arg)`.
const REVERT_TO: u8 = 3;
/// `vm.transact`-like nested transaction against `COUNTER` (arg 0) or `COLD_COUNTER` (arg 1),
/// published to the accepted overlay mid-transaction.
const TRANSACT: u8 = 4;
/// Writes `COUNTER` slot 0 = 100 through `overlay_db_mut`, as the issue prescribes for
/// `loadAllocs` and `cloneAccount`.
const OVERLAY_WRITE: u8 = 5;

/// Read-only backing database standing in for `SharedBackend`: shared by clones, `Send + Sync`,
/// and never written by commits.
#[derive(Debug, Default)]
struct Backing {
    accounts: AddressMap<AccountInfo>,
    code: B256Map<Bytecode>,
    storage: HashMap<(Address, Word), Word>,
    reads: AtomicUsize,
    fail_storage: AtomicBool,
}

#[derive(Clone, Debug)]
struct BackingDb(Arc<Backing>);

impl BackingDb {
    fn with_counter(value: u64) -> Self {
        let mut backing = Backing::default();
        for address in [COUNTER, COLD_COUNTER] {
            let info = AccountInfo::default().with_code(counter_code());
            backing.code.insert(info.code_hash, counter_code());
            backing.accounts.insert(address, info);
            backing.storage.insert((address, Word::ZERO), Word::from(value));
        }
        Self(Arc::new(backing))
    }

    fn reads(&self) -> usize {
        self.0.reads.load(Ordering::Relaxed)
    }

    fn fail_storage(&self, fail: bool) {
        self.0.fail_storage.store(fail, Ordering::Relaxed);
    }
}

impl DynDatabase for BackingDb {
    fn get_account(&mut self, address: &Address) -> DbResult<Option<AccountInfo>> {
        self.0.reads.fetch_add(1, Ordering::Relaxed);
        Ok(self.0.accounts.get(address).cloned())
    }

    fn get_code_by_hash(&mut self, code_hash: &B256) -> DbResult<Bytecode> {
        Ok(self.0.code.get(code_hash).cloned().unwrap_or_default())
    }

    fn get_storage(&mut self, address: &Address, key: &Word) -> DbResult<Word> {
        self.0.reads.fetch_add(1, Ordering::Relaxed);
        if self.0.fail_storage.load(Ordering::Relaxed) {
            return Err(DatabaseError::new(std::io::Error::other("backing read failed"), false));
        }
        Ok(self.0.storage.get(&(*address, *key)).copied().unwrap_or_default())
    }

    fn get_block_hash(&mut self, _number: &Word) -> DbResult<B256> {
        Ok(B256::ZERO)
    }
}

/// A fork's state while no [`Evm`] runs it. Unlike [`State`], it is `Send`.
#[derive(Clone, Debug)]
struct SavedState {
    /// Accepted overlay cache, moved out of the live state.
    cache: Cache,
    /// Everything else: transaction layer, journal, logs, BAL context. Captured with an empty
    /// cache, so saving never clones the cache.
    rest: StateSnapshot,
}

impl Default for SavedState {
    fn default() -> Self {
        Self::save(State::new(EmptyDB::default()))
    }
}

impl SavedState {
    /// Moves the overlay cache out of `state` and captures the rest.
    fn save(mut state: State<'_>) -> Self {
        let cache = mem::take(&mut state.overlay_db_mut().cache);
        Self { cache, rest: state.snapshot() }
    }

    /// Rebuilds a live state over `db`, moving the cache back in.
    fn load<'a>(self, db: impl DynDatabase + 'a) -> State<'a> {
        let mut state = self.rest.into_state(db);
        state.overlay_db_mut().cache = self.cache;
        state
    }

    /// Accepts the transaction layer into the overlay at the end of a committed transaction, then
    /// clears per-transaction substate.
    ///
    /// The writes can't stay pending in the transaction layer: the next transaction needs each
    /// original reset to its start value, and a commit only accepts entries that differ from their
    /// original, so a write that isn't repeated after the fork is reselected would be lost.
    fn accept_transaction(self) -> Self {
        let mut state = self.load(EmptyDB::default());
        state.commit_transaction();
        state.clear_transaction_state();
        Self::save(state)
    }

    /// Reads a slot through the transaction layer, the overlay, then `db`.
    fn storage(&self, db: BackingDb, address: Address, key: Word) -> Word {
        self.clone().load(db).storage_slot_untracked(&address, &key).unwrap()
    }

    /// Captures what [`Self::save`] keeps besides the cache, leaving `state` live.
    fn rest_of(state: &mut State<'_>) -> StateSnapshot {
        let cache = mem::take(&mut state.overlay_db_mut().cache);
        let rest = state.snapshot();
        state.overlay_db_mut().cache = cache;
        rest
    }
}

#[derive(Clone, Debug)]
struct ForkEntry {
    backing: BackingDb,
    /// `None` while the fork is active: its state then lives in the executor or the running `Evm`.
    saved: Option<SavedState>,
}

/// Like Foundry's, a snapshot covers only the fork it was taken on.
#[derive(Clone, Debug)]
struct RegistrySnapshot {
    active: usize,
    active_state: StateSnapshot,
    /// [`Cheats::selected`] when the snapshot was taken.
    selected: StateSnapshot,
}

/// What remains of Foundry's backend: no cache layer of its own.
#[derive(Clone, Debug, Default)]
struct Registry {
    forks: Vec<ForkEntry>,
    active: usize,
    snapshots: Vec<RegistrySnapshot>,
    persistent: AddressSet,
}

impl Registry {
    fn backing(&self, fork: usize) -> BackingDb {
        self.forks[fork].backing.clone()
    }
}

/// The cheatcode inspector. Owns the registry for the duration of one call.
struct Cheats {
    registry: Registry,
    /// Speculative calls capture the registry before its first mutation.
    speculative: bool,
    /// The fork active when the call started.
    start_fork: usize,
    /// The active fork's transaction layer, journal, and logs as it was selected, or as the call
    /// started. Foundry stores this with the fork as its `journaled_state`. A snapshot restore
    /// that leaves the fork puts it back, so only the fork's writes since are dropped.
    selected: StateSnapshot,
    /// Registry captured before the first mutation of a speculative call.
    captured: Option<Registry>,
    /// Whether to also capture the start fork's accepted overlay. The issue only names the
    /// registry and saved forks; without this, overlay writes and snapshot restores of the start
    /// fork survive the discard.
    capture_start_overlay: bool,
    captured_start_cache: Option<Cache>,
    /// Run depth-1 calls as isolated transactions.
    isolate: bool,
    /// Set while the inspector runs in an isolated child, whose calls aren't isolated again.
    in_child: bool,
    /// Snapshot restores in the running child that are still in effect, like master's
    /// `isolated_snapshot_restores`.
    child_restores: Vec<ChildRestore>,
    /// Calls running in the child, like master's `isolated_frame_checkpoints`.
    child_calls: Vec<ChildCall>,
    /// Whether the parent's overlay was empty while the isolated child ran.
    parent_overlay_moved: Vec<bool>,
}

/// The placeholder left in the parent while the isolated child runs the inspector.
impl Default for Cheats {
    fn default() -> Self {
        Self {
            registry: Registry::default(),
            speculative: false,
            start_fork: 0,
            selected: SavedState::default().rest,
            captured: None,
            capture_start_overlay: false,
            captured_start_cache: None,
            isolate: false,
            in_child: false,
            child_restores: Vec::new(),
            child_calls: Vec::new(),
            parent_overlay_moved: Vec::new(),
        }
    }
}

/// What a snapshot restore in an isolated child replaced.
struct ChildRestore {
    /// The fork active before the restore.
    fork: usize,
    state: StateSnapshot,
    selected: StateSnapshot,
}

/// A call running in an isolated child.
struct ChildCall {
    checkpoint: StateCheckpoint,
    /// Length of [`Cheats::child_restores`] when the call started.
    restores: usize,
}

impl Inspector<BaseEvmTypes> for Cheats {
    fn call(
        &mut self,
        interp: &mut Interpreter<'_, '_, BaseEvmTypes>,
        message: &mut Message<BaseEvmTypes>,
    ) -> Option<MessageResult<BaseEvmTypes>> {
        if message.destination == CHEATS {
            let output = self.dispatch(interp.host(), &message.input);
            return Some(message_result(message, output.is_some(), output.unwrap_or_default()));
        }
        if self.isolate && !self.in_child && message.depth == 1 {
            return Some(self.isolated_call(interp.host(), message));
        }
        if self.in_child {
            let checkpoint = interp.host().state().checkpoint();
            self.child_calls.push(ChildCall { checkpoint, restores: self.child_restores.len() });
        }
        None
    }

    fn call_end(
        &mut self,
        interp: &mut Interpreter<'_, '_, BaseEvmTypes>,
        message: &Message<BaseEvmTypes>,
        result: &mut MessageResult<BaseEvmTypes>,
    ) {
        if self.in_child && message.destination != CHEATS {
            self.finish_child_call(interp.host(), result.is_success());
        }
    }
}

impl Cheats {
    fn dispatch(&mut self, evm: &mut Evm<'_, BaseEvmTypes>, input: &[u8]) -> Option<Bytes> {
        match input[0] {
            SELECT_FORK => {
                self.capture(evm);
                self.select_fork(evm, usize::from(input[1]));
                Some(Bytes::new())
            }
            SNAPSHOT => {
                self.capture(evm);
                let snapshot = RegistrySnapshot {
                    active: self.registry.active,
                    active_state: evm.state().snapshot(),
                    selected: self.selected.clone(),
                };
                self.registry.snapshots.push(snapshot);
                Some(Bytes::new())
            }
            REVERT_TO => {
                self.capture(evm);
                if self.in_child {
                    self.child_restores.push(ChildRestore {
                        fork: self.registry.active,
                        state: evm.state().snapshot(),
                        selected: self.selected.clone(),
                    });
                }
                self.revert_to(evm, usize::from(input[1]));
                Some(Bytes::new())
            }
            TRANSACT => {
                self.capture(evm);
                let target = if input[1] == 0 { COUNTER } else { COLD_COUNTER };
                self.transact(evm, target)
            }
            OVERLAY_WRITE => {
                self.capture(evm);
                evm.overlay_db_mut().insert_account_storage(
                    &COUNTER,
                    &Word::ZERO,
                    &Word::from(100),
                );
                Some(Bytes::new())
            }
            _ => None,
        }
    }

    /// Lazily captures what a speculative call must restore.
    fn capture(&mut self, evm: &Evm<'_, BaseEvmTypes>) {
        if !self.speculative || self.captured.is_some() {
            return;
        }
        self.captured = Some(self.registry.clone());
        // The first mutation always happens while the start fork is still active.
        if self.capture_start_overlay {
            self.captured_start_cache = Some(evm.overlay_db().cache.clone());
        }
    }

    /// Saves the active fork and loads `fork` into the running `Evm`.
    ///
    /// Persistent accounts follow the switch with both layers, as in Foundry's
    /// `merge_account_data`: the accepted overlay entry, then the transaction layer. Like the
    /// fork's `journaled_state` in Foundry, [`Self::selected`] is recorded after the merge.
    fn select_fork(&mut self, evm: &mut Evm<'_, BaseEvmTypes>, fork: usize) {
        if self.registry.active == fork {
            return;
        }
        let incoming = self.registry.forks[fork].saved.take().expect("inactive fork is saved");
        let outgoing = mem::replace(evm.state_mut(), incoming.load(self.registry.backing(fork)));
        for address in &self.registry.persistent {
            merge_accepted_account(
                &mut evm.overlay_db_mut().cache,
                &outgoing.overlay_db().cache,
                address,
            );
            evm.state_mut().merge_transaction_account_from(address, &outgoing);
        }
        self.selected = SavedState::rest_of(evm.state_mut());
        let active = mem::replace(&mut self.registry.active, fork);
        self.registry.forks[active].saved = Some(SavedState::save(outgoing));
    }

    /// Restores the fork the snapshot was taken on, persistent accounts included, and leaves
    /// the other forks alone, as Foundry's `revert_state` does.
    ///
    /// If another fork is active, it keeps its accepted overlay and gets back the transaction
    /// layer it was selected with, as Foundry keeps that fork's database and `journaled_state`.
    /// Only its writes since it was selected are dropped.
    fn revert_to(&mut self, evm: &mut Evm<'_, BaseEvmTypes>, id: usize) {
        let RegistrySnapshot { active, active_state, selected } =
            self.registry.snapshots[id].clone();
        let restored = active_state.into_state(self.registry.backing(active));
        let mut left = mem::replace(evm.state_mut(), restored);
        let left_selected = mem::replace(&mut self.selected, selected);
        let left_fork = mem::replace(&mut self.registry.active, active);
        if left_fork != active {
            let cache = mem::take(&mut left.overlay_db_mut().cache);
            self.registry.forks[left_fork].saved = Some(SavedState { cache, rest: left_selected });
            self.registry.forks[active].saved = None;
        }
    }

    /// Runs a nested transaction over the moved overlay and publishes it mid-transaction.
    ///
    /// Execution is the only fallible step. Publishing writes the accepted overlay and refreshes
    /// the live transaction layer from the child's pending state, without journaling and without
    /// further reads.
    fn transact(&mut self, evm: &mut Evm<'_, BaseEvmTypes>, target: Address) -> Option<Bytes> {
        let (output, pending) = run_child(evm, CALLER, target, Bytes::new(), 1_000_000).ok()?;
        evm.overlay_db_mut().commit_pending(&pending);
        evm.state_mut().merge_isolated_state(pending);
        Some(output)
    }

    /// Runs the call as a transaction in a child [`Evm`] that runs this inspector too, so
    /// cheatcodes called inside the child are dispatched, then folds the child's state back in
    /// like master's `transact_inner`.
    fn isolated_call(
        &mut self,
        evm: &mut Evm<'_, BaseEvmTypes>,
        message: &Message<BaseEvmTypes>,
    ) -> MessageResult<BaseEvmTypes> {
        let source_fork = self.registry.active;
        let cheats = Rc::new(RefCell::new(Self { in_child: true, ..mem::take(self) }));
        let mut moved = false;
        let result = run_child_with(evm, |parent, child| {
            moved = parent.overlay_db().cache.accounts.is_empty()
                && child.overlay_db().cache.accounts.contains_key(&COUNTER);
            let tx = legacy_tx(
                message.caller,
                message.destination,
                message.input.clone(),
                message.gas_limit,
            );
            child.set_inspector(ChildCheats(Rc::clone(&cheats)));
            child.transact(&tx).map(|executed| executed.detach())
        });
        let cheats = Rc::into_inner(cheats).expect("the child is dropped").into_inner();
        *self = Self { in_child: false, ..cheats };
        let restored = !mem::take(&mut self.child_restores).is_empty();
        self.child_calls.clear();
        self.parent_overlay_moved.push(moved);
        match result {
            Ok(out) => {
                let success = out.result.status;
                if self.registry.active != source_fork || (success && restored) {
                    // The child's state replaces the parent's after a fork switch, as in master.
                    // After a snapshot restore it also drops what the child no longer has, like
                    // master's `merge_child_state(.., true)`.
                    evm.state_mut().set_pending_state(out.pending_state);
                } else {
                    evm.state_mut().merge_isolated_state(out.pending_state);
                }
                message_result(message, success, out.result.output)
            }
            Err(_) => message_result(message, false, Bytes::new()),
        }
    }

    /// Undoes the snapshot restores of a failed call in the child, like master's
    /// `finish_isolated_snapshot_frame`, so they don't escape the call.
    ///
    /// evm2 already rolled the call back, but on the restored journal, which the call's checkpoint
    /// doesn't index. Put back the state from before the call's first restore and roll that back.
    /// Like fork selections, restores that end on another fork are not undone.
    fn finish_child_call(&mut self, evm: &mut Evm<'_, BaseEvmTypes>, success: bool) {
        let call = self.child_calls.pop().expect("the call started in the child");
        if success || self.child_restores.len() == call.restores {
            return;
        }
        let ChildRestore { fork, state, selected } =
            self.child_restores.drain(call.restores..).next().unwrap();
        if fork != self.registry.active {
            return;
        }
        let mut state = state.into_state(self.registry.backing(fork));
        state.rollback(call.checkpoint, evm.version().features);
        *evm.state_mut() = state;
        self.selected = selected;
    }
}

/// Runs the parent's [`Cheats`] in an isolated child. [`Evm::clear_inspector_as`] needs a
/// `'static` [`Evm`], so the parent takes the inspector back through the shared cell.
struct ChildCheats(Rc<RefCell<Cheats>>);

impl Inspector<BaseEvmTypes> for ChildCheats {
    fn call(
        &mut self,
        interp: &mut Interpreter<'_, '_, BaseEvmTypes>,
        message: &mut Message<BaseEvmTypes>,
    ) -> Option<MessageResult<BaseEvmTypes>> {
        self.0.borrow_mut().call(interp, message)
    }

    fn call_end(
        &mut self,
        interp: &mut Interpreter<'_, '_, BaseEvmTypes>,
        message: &Message<BaseEvmTypes>,
        result: &mut MessageResult<BaseEvmTypes>,
    ) {
        self.0.borrow_mut().call_end(interp, message, result);
    }
}

/// Runs `f` with a child [`Evm`] that owns the parent's overlay, then moves the overlay back.
///
/// The child's transaction layer starts from the parent's via `prepare_isolated_state`.
fn run_child_with<'a, R>(
    parent: &mut Evm<'a, BaseEvmTypes>,
    f: impl FnOnce(&Evm<'a, BaseEvmTypes>, &mut Evm<'a, BaseEvmTypes>) -> R,
) -> R {
    let mut child = new_evm(EmptyDB::default());
    mem::swap(child.overlay_db_mut(), parent.overlay_db_mut());
    child.state_mut().set_pending_state(parent.state().prepare_isolated_state());
    let result = f(parent, &mut child);
    mem::swap(child.overlay_db_mut(), parent.overlay_db_mut());
    result
}

fn run_child(
    parent: &mut Evm<'_, BaseEvmTypes>,
    caller: Address,
    to: Address,
    input: Bytes,
    gas_limit: u64,
) -> Result<(Bytes, PendingState), ()> {
    run_child_with(parent, |_, child| {
        let out = child.transact(&legacy_tx(caller, to, input, gas_limit)).map_err(drop)?.detach();
        if out.result.status { Ok((out.result.output, out.pending_state)) } else { Err(()) }
    })
}

/// Merges the accepted entry of `address` in `source`, if cached, into `target`, giving it
/// precedence as [`Cache::merge`] does: account, code, and storage, including a wipe.
///
/// [`State::merge_transaction_account_from`] alone is not enough. It skips accounts not loaded in
/// this transaction, and the original values it copies come from `source`. A commit only accepts
/// entries that differ from their original, so an entry that is loaded but unchanged would fall
/// back to `target`'s own state once the transaction ends.
fn merge_accepted_account(target: &mut Cache, source: &Cache, address: &Address) {
    let mut entry = Cache::default();
    if let Some(account) = source.accounts.get(address) {
        if let Some(info) = account
            && let Some(code) = source.contracts.get(&info.code_hash)
        {
            entry.contracts.insert(info.code_hash, code.clone());
        }
        entry.accounts.insert(*address, account.clone());
    }
    if let Some(storage) = source.storage.get(address) {
        entry.storage.insert(*address, storage.clone());
    }
    target.merge(entry);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Commit,
    Discard,
}

/// Foundry-style executor: owns the registry and the active fork's state between calls.
#[derive(Clone)]
struct Executor {
    registry: Registry,
    /// The active fork's state while no call runs.
    active: Option<SavedState>,
    capture_start_overlay: bool,
    isolate: bool,
    parent_overlay_moved: Vec<bool>,
}

impl Executor {
    fn new(backings: Vec<BackingDb>) -> Self {
        let forks = backings
            .into_iter()
            .map(|backing| ForkEntry { backing, saved: Some(SavedState::default()) })
            .collect();
        let mut registry = Registry { forks, ..Default::default() };
        registry.persistent.extend([CALLER, TEST]);
        let active = registry.forks[0].saved.take();
        Self {
            registry,
            active,
            capture_start_overlay: true,
            isolate: false,
            parent_overlay_moved: Vec::new(),
        }
    }

    /// Writes outside a transaction go straight to the active overlay.
    fn set_code(&mut self, address: Address, code: Bytecode) {
        let mut state = self.active.take().unwrap().load(EmptyDB::default());
        state
            .overlay_db_mut()
            .insert_account_info(&address, AccountInfo::default().with_code(code));
        self.active = Some(SavedState::save(state));
    }

    /// Calls the test contract running `script`.
    fn run(&mut self, script: &[(Address, &[u8])], mode: Mode) {
        self.set_code(TEST, script_code(script));
        let active = self.registry.active;
        let mut evm = new_evm(EmptyDB::default());
        let state = self.active.take().unwrap();
        // The previous call accepted or dropped the active fork's transaction layer, so the call
        // start counts as its selection.
        let selected = state.rest.clone();
        *evm.state_mut() = state.load(self.registry.backing(active));
        evm.set_inspector(Cheats {
            registry: mem::take(&mut self.registry),
            speculative: mode == Mode::Discard,
            start_fork: active,
            selected,
            capture_start_overlay: self.capture_start_overlay,
            isolate: self.isolate,
            ..Default::default()
        });
        let executed = evm.transact(&legacy_tx(CALLER, TEST, Bytes::new(), 10_000_000)).unwrap();
        let result = match mode {
            Mode::Commit => executed.commit(),
            Mode::Discard => executed.discard(),
        };
        let cheats = *evm.clear_inspector_as::<Cheats>().unwrap();
        let state = mem::replace(evm.state_mut(), State::new(EmptyDB::default()));
        self.finish(cheats, state, mode);
        assert!(result.status, "test call failed: {result:?}");
    }

    fn finish(&mut self, cheats: Cheats, state: State<'_>, mode: Mode) {
        let Cheats { mut registry, start_fork, captured, captured_start_cache, .. } = cheats;
        self.parent_overlay_moved.extend(cheats.parent_overlay_moved);
        if let Some(captured) = captured {
            // Speculative call that mutated the registry: restore it and recover the start fork.
            let mut start = if registry.active == start_fork {
                state
            } else {
                registry.forks[start_fork].saved.take().unwrap().load(EmptyDB::default())
            };
            start.clear_transaction_state();
            if let Some(cache) = captured_start_cache {
                start.overlay_db_mut().cache = cache;
            }
            self.active = Some(SavedState::save(start));
            self.registry = captured;
            return;
        }
        if mode == Mode::Commit {
            // `ExecutedTx::commit` only accepted the active fork; accept the inactive ones too.
            for fork in &mut registry.forks {
                fork.saved = fork.saved.take().map(SavedState::accept_transaction);
            }
        }
        self.active = Some(SavedState::save(state));
        self.registry = registry;
    }

    /// Reads a slot of `fork` the way executor helpers do: through the overlay and saved caches.
    fn storage(&self, fork: usize, address: Address, key: Word) -> Word {
        let saved = if fork == self.registry.active {
            self.active.as_ref()
        } else {
            self.registry.forks[fork].saved.as_ref()
        };
        saved.unwrap().storage(self.registry.backing(fork), address, key)
    }

    /// The accepted overlay's cached value of a slot, if any.
    fn accepted(&self, fork: usize, address: Address, key: Word) -> Option<Word> {
        let saved = if fork == self.registry.active {
            self.active.as_ref()
        } else {
            self.registry.forks[fork].saved.as_ref()
        };
        saved.unwrap().cache.storage.get(&address)?.slots.get(&key).copied()
    }
}

fn execution_config() -> ExecutionConfig<BaseEvmTypes> {
    let mut version = Version::new(SPEC);
    // Foundry disables these for test execution.
    version.features.remove(
        EvmFeatures::NONCE_CHECK | EvmFeatures::EIP3607 | EvmFeatures::BLOCK_GAS_LIMIT_CHECK,
    );
    ExecutionConfig::for_spec_and_version(SPEC, version)
}

fn new_evm<'a>(db: impl DynDatabase + 'a) -> Evm<'a, BaseEvmTypes> {
    Evm::new_with_execution_config(
        execution_config(),
        SPEC,
        BlockEnvExt::default(),
        ethereum_tx_registry(SPEC),
        db,
        Precompiles::base(SPEC),
    )
}

fn legacy_tx(caller: Address, to: Address, input: Bytes, gas_limit: u64) -> Recovered<TxEnvelope> {
    Recovered::new_unchecked(
        TxEnvelope::Legacy(TxLegacy {
            to: TxKind::Call(to),
            input,
            gas_limit,
            ..Default::default()
        }),
        caller,
    )
}

fn message_result(
    message: &Message<BaseEvmTypes>,
    success: bool,
    output: Bytes,
) -> MessageResult<BaseEvmTypes> {
    MessageResult::<BaseEvmTypes> {
        stop: if success { InstrStop::Return } else { InstrStop::Revert },
        gas: GasTracker::new(message.gas_limit),
        output,
        created_address: None,
        ext: Default::default(),
        _non_exhaustive: (),
    }
}

/// Increments slot 0 and returns the new value.
fn counter_code() -> Bytecode {
    Bytecode::new_legacy(Bytes::from_static(&[
        op::PUSH0,
        op::SLOAD,
        op::PUSH1,
        1,
        op::ADD,
        op::DUP1,
        op::PUSH0,
        op::SSTORE,
        op::PUSH0,
        op::MSTORE,
        op::PUSH1,
        32,
        op::PUSH0,
        op::RETURN,
    ]))
}

/// Code that performs `calls` in order and ignores their results.
fn script_code(calls: &[(Address, &[u8])]) -> Bytecode {
    script_code_ending(calls, &[op::STOP])
}

/// Like [`script_code`], then reverts.
fn reverting_script_code(calls: &[(Address, &[u8])]) -> Bytecode {
    script_code_ending(calls, &[op::PUSH0, op::PUSH0, op::REVERT])
}

fn script_code_ending(calls: &[(Address, &[u8])], end: &[u8]) -> Bytecode {
    let mut code = Vec::new();
    for (to, data) in calls {
        let mut word = [0; 32];
        word[..data.len()].copy_from_slice(data);
        code.push(op::PUSH32);
        code.extend_from_slice(&word);
        code.extend([op::PUSH0, op::MSTORE]);
        // CALL(gas, to, value, argsOffset, argsSize, retOffset, retSize), pushed in reverse.
        code.extend([op::PUSH0, op::PUSH0, op::PUSH1, data.len() as u8, op::PUSH0, op::PUSH0]);
        code.push(op::PUSH20);
        code.extend_from_slice(to.as_slice());
        code.extend([op::GAS, op::CALL, op::POP]);
    }
    code.extend_from_slice(end);
    Bytecode::new_legacy(code.into())
}

/// An executor over two forks, with counters at 10 and 20, and `contracts` on the first.
fn executor_with(contracts: &[(Address, Bytecode)]) -> Executor {
    let mut executor =
        Executor::new(vec![BackingDb::with_counter(10), BackingDb::with_counter(20)]);
    for (address, code) in contracts {
        executor.set_code(*address, code.clone());
    }
    executor
}

fn counter(executor: &Executor, fork: usize) -> u64 {
    executor.storage(fork, COUNTER, Word::ZERO).to::<u64>()
}

const INC: (Address, &[u8]) = (COUNTER, &[]);

const fn select(fork: u8) -> (Address, &'static [u8]) {
    match fork {
        0 => (CHEATS, &[SELECT_FORK, 0]),
        _ => (CHEATS, &[SELECT_FORK, 1]),
    }
}

#[test]
fn commit_on_active_fork() {
    let backing = BackingDb::with_counter(10);
    let mut executor = Executor::new(vec![backing.clone()]);

    executor.run(&[INC], Mode::Commit);
    assert_eq!(executor.accepted(0, COUNTER, Word::ZERO), Some(Word::from(11)));
    let reads = backing.reads();

    executor.run(&[INC, INC], Mode::Commit);
    assert_eq!(counter(&executor, 0), 13);
    // The overlay serves the second transaction: the counter is not fetched again.
    assert_eq!(backing.reads(), reads);

    executor.run(&[INC], Mode::Discard);
    assert_eq!(counter(&executor, 0), 13);
    // Commits never write the backing database.
    assert_eq!(backing.0.storage[&(COUNTER, Word::ZERO)], Word::from(10));
}

#[test]
fn save_and_reload_fork_with_pending_writes_across_transactions() {
    let a = BackingDb::with_counter(10);
    let mut executor = Executor::new(vec![a.clone(), BackingDb::with_counter(20)]);

    // Write on A, switch to B mid-transaction, write on B, commit while B is active.
    executor.run(&[INC, select(1), INC], Mode::Commit);
    assert_eq!(executor.registry.active, 1);
    assert_eq!(executor.accepted(1, COUNTER, Word::ZERO), Some(Word::from(21)));
    // The commit also accepts A's write into A's own overlay, though A is inactive.
    assert_eq!(executor.accepted(0, COUNTER, Word::ZERO), Some(Word::from(11)));
    assert_eq!(counter(&executor, 0), 11);
    // Its per-transaction substate was cleared at the boundary.
    let saved = executor.registry.forks[0].saved.clone().unwrap().load(EmptyDB::default());
    assert!(saved.journal().is_empty());

    // A later transaction reloads A with its accepted write and leaves it inactive again.
    let reads = a.reads();
    executor.run(&[select(0), INC, select(1), INC], Mode::Commit);
    assert_eq!((counter(&executor, 0), counter(&executor, 1)), (12, 22));
    assert_eq!(a.reads(), reads, "reloading A must reuse its moved cache");

    // Committing while A is active accepts its write as usual.
    executor.run(&[select(0), INC], Mode::Commit);
    assert_eq!(executor.accepted(0, COUNTER, Word::ZERO), Some(Word::from(13)));
    assert_eq!((counter(&executor, 0), counter(&executor, 1)), (13, 22));
}

/// Writes on a fork that is inactive at commit must not depend on being repeated once it is
/// reselected, as they would if they stayed pending in its transaction layer.
#[test]
fn reselected_fork_keeps_writes_it_does_not_repeat() {
    let counters = |executor: &Executor| {
        (counter(executor, 0), executor.storage(0, COLD_COUNTER, Word::ZERO).to::<u64>())
    };
    let mut executor =
        Executor::new(vec![BackingDb::with_counter(10), BackingDb::with_counter(20)]);
    // Write two slots on A, then commit while B is active.
    executor.run(&[INC, (COLD_COUNTER, &[]), select(1)], Mode::Commit);

    // Committing on A without writing keeps both.
    let mut reselected = executor.clone();
    reselected.run(&[select(0)], Mode::Commit);
    assert_eq!(reselected.registry.active, 0);
    assert_eq!(counters(&reselected), (11, 11));

    // Rewriting one keeps the other.
    executor.run(&[select(0), INC], Mode::Commit);
    assert_eq!(counters(&executor), (12, 11));
}

/// A persistent account follows a fork switch even when the transaction hasn't loaded it, as when
/// `setUp` committed it.
#[test]
fn persistent_account_follows_fork_switch() {
    let mut executor =
        Executor::new(vec![BackingDb::with_counter(10), BackingDb::with_counter(20)]);
    executor.registry.persistent.insert(COUNTER);
    executor.run(&[INC], Mode::Commit);

    executor.run(&[select(1), INC], Mode::Commit);
    assert_eq!(counter(&executor, 1), 12);
    executor.run(&[select(0), INC], Mode::Commit);
    assert_eq!(counter(&executor, 0), 13);
}

/// A persistent account that the transaction loads but leaves unchanged keeps its state on the
/// fork it was switched to, even though the commit doesn't accept it there.
#[test]
fn persistent_account_unchanged_by_transaction_survives_commit() {
    let deployed = |executor: &Executor| executor.storage(1, DEPLOYED, Word::ZERO).to::<u64>();
    let mut executor =
        Executor::new(vec![BackingDb::with_counter(10), BackingDb::with_counter(20)]);
    executor.set_code(DEPLOYED, counter_code());
    executor.registry.persistent.insert(DEPLOYED);

    // Both calls change the slot, but none changes the account with the code.
    executor.run(&[(DEPLOYED, &[]), select(1), (DEPLOYED, &[])], Mode::Commit);
    assert_eq!(deployed(&executor), 2);
    executor.run(&[(DEPLOYED, &[])], Mode::Commit);
    assert_eq!(deployed(&executor), 3);
}

/// Like Foundry's `revert_state`, a snapshot restore also reverts persistent accounts.
#[test]
fn snapshot_restore_reverts_persistent_accounts() {
    let mut executor = Executor::new(vec![BackingDb::with_counter(10)]);
    executor.registry.persistent.insert(COUNTER);
    executor.run(&[(CHEATS, &[SNAPSHOT]), INC, (CHEATS, &[REVERT_TO, 0])], Mode::Commit);
    assert_eq!(counter(&executor, 0), 10);
}

/// Like Foundry's, a snapshot restore replaces only the fork the snapshot was taken on. Other
/// forks keep their committed and pending writes.
#[test]
fn snapshot_restore_leaves_other_forks() {
    let new = || Executor::new(vec![BackingDb::with_counter(10), BackingDb::with_counter(20)]);
    let counters = |executor: &Executor| (counter(executor, 0), counter(executor, 1));

    // B's write was committed while B was active, and B is still active at the restore.
    let mut executor = new();
    executor.run(&[(CHEATS, &[SNAPSHOT])], Mode::Commit);
    executor.run(&[INC, select(1), INC], Mode::Commit);
    executor.run(&[(CHEATS, &[REVERT_TO, 0])], Mode::Commit);
    assert_eq!(executor.registry.active, 0);
    assert_eq!(counters(&executor), (10, 21));

    // B's write was committed while B was inactive.
    let mut executor = new();
    executor.run(&[(CHEATS, &[SNAPSHOT])], Mode::Commit);
    executor.run(&[INC, select(1), INC, select(0)], Mode::Commit);
    executor.run(&[(CHEATS, &[REVERT_TO, 0])], Mode::Commit);
    assert_eq!(counters(&executor), (10, 21));

    // B's write is still pending in its saved state when A's snapshot is restored.
    let mut executor = new();
    let script: &[(Address, &[u8])] =
        &[(CHEATS, &[SNAPSHOT]), INC, select(1), INC, select(0), (CHEATS, &[REVERT_TO, 0])];
    executor.run(script, Mode::Commit);
    assert_eq!(counters(&executor), (10, 21));
}

/// A restore that leaves the active fork drops only what the fork wrote since it was selected, as
/// Foundry keeps that fork's database and the `journaled_state` it was selected with.
#[test]
fn snapshot_restore_drops_writes_since_left_fork_was_selected() {
    let new = || Executor::new(vec![BackingDb::with_counter(10), BackingDb::with_counter(20)]);

    // B was selected in this transaction.
    let mut executor = new();
    executor.run(&[(CHEATS, &[SNAPSHOT]), select(1), INC, (CHEATS, &[REVERT_TO, 0])], Mode::Commit);
    assert_eq!(executor.registry.active, 0);
    assert_eq!(counter(&executor, 1), 20);

    // B was selected in an earlier call.
    let mut executor = new();
    executor.run(&[(CHEATS, &[SNAPSHOT]), select(1)], Mode::Commit);
    executor.run(&[INC, (CHEATS, &[REVERT_TO, 0])], Mode::Commit);
    assert_eq!(counter(&executor, 1), 20);

    // B's write from an earlier selection stays pending; only the one since it was reselected is
    // dropped.
    let mut executor = new();
    let script: &[(Address, &[u8])] = &[
        (CHEATS, &[SNAPSHOT]),
        select(1),
        INC,
        select(0),
        select(1),
        INC,
        (CHEATS, &[REVERT_TO, 0]),
    ];
    executor.run(script, Mode::Commit);
    assert_eq!(counter(&executor, 1), 21);

    // `vm.transact` writes B's overlay, as Foundry writes the fork's database, so it stays.
    let mut executor = new();
    let script: &[(Address, &[u8])] =
        &[(CHEATS, &[SNAPSHOT]), select(1), (CHEATS, &[TRANSACT, 0]), (CHEATS, &[REVERT_TO, 0])];
    executor.run(script, Mode::Commit);
    assert_eq!(counter(&executor, 1), 21);
}

#[test]
fn speculative_call_restores_registry_after_fork_switch_and_snapshot_restore() {
    let mut executor =
        Executor::new(vec![BackingDb::with_counter(10), BackingDb::with_counter(20)]);
    executor.run(&[INC, (CHEATS, &[SNAPSHOT])], Mode::Commit);
    assert_eq!(executor.registry.snapshots.len(), 1);

    let script: &[(Address, &[u8])] = &[
        INC,
        select(1),
        INC,
        (CHEATS, &[SNAPSHOT]),
        INC,
        (CHEATS, &[REVERT_TO, 1]),
        select(0),
        INC,
    ];
    // Control: committed, the script switches forks, snapshots, and restores as intended.
    let mut control = executor.clone();
    control.run(script, Mode::Commit);
    assert_eq!(control.registry.snapshots.len(), 2);
    assert_eq!((counter(&control, 0), counter(&control, 1)), (13, 21));

    executor.run(script, Mode::Discard);
    assert_eq!(executor.registry.active, 0);
    assert_eq!(executor.registry.snapshots.len(), 1);
    assert_eq!((counter(&executor, 0), counter(&executor, 1)), (11, 20));
}

/// Gap: capturing only the registry and saved forks, as the issue says, is not enough.
#[test]
fn speculative_call_without_start_overlay_capture_leaks() {
    // An overlay write on the start fork survives the discard.
    let mut executor = Executor::new(vec![BackingDb::with_counter(10)]);
    executor.capture_start_overlay = false;
    executor.run(&[(CHEATS, &[TRANSACT, 0])], Mode::Discard);
    assert_eq!(counter(&executor, 0), 11, "vm.transact in a test leaked into the next test");

    // Restoring an older snapshot replaces the start fork's accepted overlay.
    let mut executor = Executor::new(vec![BackingDb::with_counter(10)]);
    executor.capture_start_overlay = false;
    executor.run(&[(CHEATS, &[SNAPSHOT])], Mode::Commit);
    executor.run(&[INC], Mode::Commit);
    executor.run(&[(CHEATS, &[REVERT_TO, 0])], Mode::Discard);
    assert_eq!(counter(&executor, 0), 10, "setUp's committed write was lost");

    // Capturing the start fork's overlay at the first mutation fixes both.
    let mut executor = Executor::new(vec![BackingDb::with_counter(10)]);
    executor.run(&[(CHEATS, &[SNAPSHOT])], Mode::Commit);
    executor.run(&[INC], Mode::Commit);
    executor.run(&[(CHEATS, &[TRANSACT, 0]), (CHEATS, &[REVERT_TO, 0])], Mode::Discard);
    assert_eq!(counter(&executor, 0), 11);
}

#[test]
fn isolated_child_shares_overlay_by_move() {
    let backing = BackingDb::with_counter(10);
    let mut executor = Executor::new(vec![backing.clone()]);
    executor.run(&[INC], Mode::Commit);
    executor.isolate = true;
    let reads = backing.reads();

    executor.run(&[INC, INC], Mode::Commit);
    // The second child starts from the first child's merged write.
    assert_eq!(counter(&executor, 0), 13);
    assert_eq!(executor.parent_overlay_moved, [true, true]);
    assert_eq!(backing.reads(), reads, "children read the moved overlay, not the backing");
}

/// Cheatcodes that a contract calls inside an isolated child reach the inspector. A fork the child
/// selects stays selected in the parent, which adopts the child's state.
#[test]
fn isolated_child_dispatches_cheatcodes() {
    for isolate in [false, true] {
        let mut executor =
            executor_with(&[(HANDLER, script_code(&[(CHEATS, &[SNAPSHOT]), select(1), INC]))]);
        executor.isolate = isolate;

        executor.run(&[(HANDLER, &[]), INC], Mode::Commit);
        assert_eq!(executor.registry.snapshots.len(), 1, "isolate: {isolate}");
        assert_eq!(executor.registry.active, 1, "isolate: {isolate}");
        assert_eq!((counter(&executor, 0), counter(&executor, 1)), (10, 22), "isolate: {isolate}");
    }
}

/// After a snapshot restore inside an isolated child, the parent drops its writes since the
/// snapshot, as master's `merge_child_state(.., remove_absent = true)` does. evm2's
/// `merge_isolated_state` has no such mode, but `set_pending_state` gives the same result: the
/// restored state's originals are the parent's own, from when it took the snapshot.
#[test]
fn isolated_child_restore_drops_parent_writes_since_snapshot() {
    for isolate in [false, true] {
        let mut executor = executor_with(&[(
            HANDLER,
            script_code(&[(CHEATS, &[REVERT_TO, 0]), (COLD_COUNTER, &[])]),
        )]);
        executor.isolate = isolate;

        executor.run(&[(CHEATS, &[SNAPSHOT]), INC, (HANDLER, &[])], Mode::Commit);
        assert_eq!(counter(&executor, 0), 10, "isolate: {isolate}");
        assert_eq!(
            executor.storage(0, COLD_COUNTER, Word::ZERO),
            Word::from(11),
            "isolate: {isolate}"
        );
    }
}

/// A restore inside an isolated child that then reverts doesn't escape it, as in master's
/// `test_reverted_isolated_restore_does_not_escape`.
#[test]
fn reverted_restore_in_isolated_child_does_not_escape() {
    let mut executor =
        executor_with(&[(FAILING, reverting_script_code(&[INC, (CHEATS, &[REVERT_TO, 0])]))]);
    executor.isolate = true;

    executor.run(&[INC, (CHEATS, &[SNAPSHOT]), INC, (FAILING, &[])], Mode::Commit);
    assert_eq!(counter(&executor, 0), 12);
}

/// A reverted call inside an isolated child takes its restore with it, and the child goes on from
/// the state before the call, as in master's `test_caught_nested_restore_revert_does_not_escape`.
#[test]
fn caught_reverted_restore_in_isolated_child_is_undone() {
    let mut executor = executor_with(&[
        (HANDLER, script_code(&[(FAILING, &[]), INC])),
        (FAILING, reverting_script_code(&[INC, (CHEATS, &[REVERT_TO, 0])])),
    ]);
    executor.isolate = true;

    executor.run(&[INC, (CHEATS, &[SNAPSHOT]), INC, (HANDLER, &[])], Mode::Commit);
    assert_eq!(counter(&executor, 0), 13);
}

/// A reverted call doesn't undo a restore made before it in the same isolated child, as in master's
/// `test_successful_restore_survives_reverted_sibling`.
#[test]
fn isolated_child_restore_survives_reverted_sibling() {
    let mut executor = executor_with(&[
        (HANDLER, script_code(&[(NESTED, &[]), INC, (FAILING, &[])])),
        (NESTED, script_code(&[INC, (CHEATS, &[SNAPSHOT]), INC, INC, (CHEATS, &[REVERT_TO, 0])])),
        (FAILING, reverting_script_code(&[])),
    ]);
    executor.isolate = true;

    // 11 after the restore, which undid 13.
    executor.run(&[(HANDLER, &[])], Mode::Commit);
    assert_eq!(counter(&executor, 0), 12);
}

/// Gap: state captured inside an isolated child loses the parent's earlier writes in the
/// transaction once it replaces the parent's state. `prepare_isolated_state` makes the child's
/// originals the parent's current values, as the child's gas accounting needs, and `Cache::commit`
/// only accepts entries that differ from their original. Master keeps the writes because its commit
/// takes the present value of every slot of a touched account. With the public API, a test can't
/// rebase such state onto the parent's originals, so evm2 needs one.
#[test]
fn state_captured_in_isolated_child_loses_parent_writes() {
    // The parent restores a snapshot taken inside a child.
    let restored = [false, true].map(|isolate| {
        let mut executor = executor_with(&[(HANDLER, script_code(&[(CHEATS, &[SNAPSHOT])]))]);
        executor.isolate = isolate;
        executor.run(&[INC, (HANDLER, &[]), INC, (CHEATS, &[REVERT_TO, 0])], Mode::Commit);
        counter(&executor, 0)
    });
    assert_eq!(restored, [11, 10], "plain, isolated");

    // A child leaves the fork, which saves the child's state for it.
    let left = [false, true].map(|isolate| {
        let mut executor = executor_with(&[(HANDLER, script_code(&[select(1)]))]);
        executor.isolate = isolate;
        executor.run(&[INC, (HANDLER, &[])], Mode::Commit);
        counter(&executor, 0)
    });
    assert_eq!(left, [11, 10], "plain, isolated");
}

#[test]
fn staged_write_mid_transaction() {
    let backing = BackingDb::with_counter(10);
    let mut executor = Executor::new(vec![backing.clone()]);

    // The nested transaction sees the in-flight write and is accepted at once; the live layer is
    // refreshed, so the next increment continues from it.
    executor.run(&[INC, (CHEATS, &[TRANSACT, 0]), INC], Mode::Commit);
    assert_eq!(counter(&executor, 0), 13);

    // A failed read inside the nested transaction publishes nothing.
    executor.run(&[INC], Mode::Commit);
    backing.fail_storage(true);
    executor.run(&[(CHEATS, &[TRANSACT, 1])], Mode::Commit);
    backing.fail_storage(false);
    assert_eq!(executor.storage(0, COLD_COUNTER, Word::ZERO), Word::from(10));
    assert_eq!(executor.accepted(0, COLD_COUNTER, Word::ZERO), None);
    assert_eq!(counter(&executor, 0), 14);

    // Control: the same nested transaction publishes once reads succeed.
    executor.run(&[(CHEATS, &[TRANSACT, 1])], Mode::Commit);
    assert_eq!(executor.accepted(0, COLD_COUNTER, Word::ZERO), Some(Word::from(11)));
}

#[test]
#[ignore = "measurement"]
fn measure_per_call_evm_construction_and_cache_move() {
    const ITERS: u32 = 20_000;

    fn time(iters: u32, mut f: impl FnMut()) -> Duration {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        start.elapsed() / iters
    }

    let backing = BackingDb::with_counter(10);
    println!(
        "Evm::new (config + tx registry + precompiles): {:?}",
        time(ITERS, || {
            black_box(new_evm(EmptyDB::default()));
        })
    );
    println!(
        "  ExecutionConfig: {:?}",
        time(ITERS, || {
            black_box(execution_config());
        })
    );
    println!(
        "  ethereum_tx_registry: {:?}",
        time(ITERS, || drop(black_box(ethereum_tx_registry::<BaseEvmTypes>(SPEC))))
    );
    println!(
        "  Precompiles::base: {:?}",
        time(ITERS, || drop(black_box(Precompiles::<BaseEvmTypes>::base(SPEC))))
    );

    for accounts in [0usize, 1_000, 100_000] {
        let mut state = State::new(EmptyDB::default());
        for i in 0..accounts {
            let address = Address::with_last_byte(0).create(i as u64);
            state.overlay_db_mut().insert_account_info(&address, AccountInfo::default());
            state.overlay_db_mut().insert_account_storage(&address, &Word::ZERO, &Word::from(i));
        }
        let mut saved = Some(SavedState::save(state));
        let moved = time(ITERS, || {
            let mut evm = new_evm(EmptyDB::default());
            *evm.state_mut() = saved.take().unwrap().load(backing.clone());
            let state = mem::replace(evm.state_mut(), State::new(EmptyDB::default()));
            saved = Some(SavedState::save(state));
        });
        let call = time(ITERS, || {
            let mut evm = new_evm(EmptyDB::default());
            *evm.state_mut() = saved.take().unwrap().load(backing.clone());
            let tx = legacy_tx(CALLER, COUNTER, Bytes::new(), 100_000);
            assert!(evm.transact(&tx).unwrap().commit().status);
            let state = mem::replace(evm.state_mut(), State::new(EmptyDB::default()));
            saved = Some(SavedState::save(state));
        });
        let loaded = saved.take().unwrap().load(backing.clone());
        let cloned = time(ITERS.min(200), || drop(black_box(loaded.snapshot())));
        println!(
            "cache of {accounts} accounts + slots: Evm::new + move in/out {moved:?}; \
             with a counter tx {call:?}; StateSnapshot clone {cloned:?}"
        );
    }
}

/// Gap: `loadAllocs` and `cloneAccount` are cheatcodes, so they run mid-transaction. A write
/// through `overlay_db_mut` is then shadowed by the transaction layer and overwritten at commit.
#[test]
fn overlay_write_mid_transaction_is_shadowed() {
    let mut executor = Executor::new(vec![BackingDb::with_counter(10)]);
    executor.run(&[INC, (CHEATS, &[OVERLAY_WRITE]), INC], Mode::Commit);
    assert_eq!(counter(&executor, 0), 12, "the cheatcode write was lost");

    // Control: the same write is visible when the slot was not loaded yet in this transaction.
    let mut executor = Executor::new(vec![BackingDb::with_counter(10)]);
    executor.run(&[(CHEATS, &[OVERLAY_WRITE]), INC], Mode::Commit);
    assert_eq!(counter(&executor, 0), 101);
}

/// The active fork's accepted overlay, borrowed read-only by a `&self` speculative call.
struct AcceptedView<'a> {
    cache: &'a Cache,
    backing: BackingDb,
}

impl DynDatabase for AcceptedView<'_> {
    fn get_account(&mut self, address: &Address) -> DbResult<Option<AccountInfo>> {
        match self.cache.accounts.get(address) {
            Some(account) => Ok(account.clone()),
            None => self.backing.get_account(address),
        }
    }

    fn get_code_by_hash(&mut self, code_hash: &B256) -> DbResult<Bytecode> {
        match self.cache.contracts.get(code_hash) {
            Some(code) => Ok(code.clone()),
            None => self.backing.get_code_by_hash(code_hash),
        }
    }

    fn get_storage(&mut self, address: &Address, key: &Word) -> DbResult<Word> {
        if let Some(storage) = self.cache.storage.get(address) {
            if let Some(value) = storage.slots.get(key) {
                return Ok(*value);
            }
            if storage.wiped {
                return Ok(Word::ZERO);
            }
        }
        if matches!(self.cache.accounts.get(address), Some(None)) {
            return Ok(Word::ZERO);
        }
        self.backing.get_storage(address, key)
    }

    fn get_block_hash(&mut self, number: &Word) -> DbResult<B256> {
        match self.cache.block_hashes.get(number) {
            Some(hash) => Ok(*hash),
            None => self.backing.get_block_hash(number),
        }
    }
}

/// Master's speculative calls take `&self` and borrow the backend (`CowBackend::new_borrowed`).
/// Moving the overlay in and out needs `&mut self`; a borrowed view keeps `&self` without a clone
/// until the first cheatcode mutation, which materializes the overlay like `CowBackend`.
#[test]
fn speculative_call_borrows_accepted_overlay() {
    let backing = BackingDb::with_counter(10);
    let mut executor = Executor::new(vec![backing.clone()]);
    executor.run(&[INC], Mode::Commit);
    let executor = &executor;
    let accepted = &executor.active.as_ref().unwrap().cache;
    let reads = backing.reads();

    let mut evm = new_evm(AcceptedView { cache: accepted, backing: backing.clone() });
    let tx = legacy_tx(CALLER, COUNTER, Bytes::new(), 100_000);
    let result = evm.transact(&tx).unwrap().discard();
    assert_eq!(Word::from_be_slice(&result.output), Word::from(12));
    assert_eq!(backing.reads(), reads, "served from the borrowed overlay");

    // First mutation: materialize the borrowed overlay, after which the state no longer needs it.
    let fills = mem::replace(&mut evm.overlay_db_mut().cache, accepted.clone());
    evm.overlay_db_mut().cache.merge(fills);
    evm.overlay_db_mut().db = Box::new(backing.clone());
    let detached = SavedState::save(mem::replace(evm.state_mut(), State::new(EmptyDB::default())));
    drop(evm);
    assert_eq!(detached.storage(backing, COUNTER, Word::ZERO), Word::from(11));
}

#[test]
fn registry_and_saved_state_are_send() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SavedState>();
    assert_send_sync::<Registry>();
    assert_send_sync::<Executor>();
}
