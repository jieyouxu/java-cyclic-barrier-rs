//! # Notice Retained from the JDK Source
//!
//! This file is available under and governed by the GNU General Public License version 2 only, as
//! published by the Free Software Foundation. However, the following notice accompanied the
//! original version of this file:
//!
//! Written by Doug Lea with assistance from members of JCP JSR-166
//! Expert Group and released to the public domain, as explained at
//! http://creativecommons.org/publicdomain/zero/1.0/

#![deny(missing_docs)]

use parking_lot::{Condvar, Mutex, ReentrantMutex, ReentrantMutexGuard};
use std::cell::RefCell;
use std::time::Duration;

/// A `BarrierCommand` is run by the last thread when reaching the common barrier point.
pub trait BarrierCommand {
    /// Error type to the barrier command function.
    type Error;

    /// What to execute when the common barrier point is reached. This is typically used to
    /// synchronize some sort of shared mutable state between threads.
    fn run(&self) -> Result<(), Self::Error>;
}

/// A no-op `BarrierCommand`.
pub struct NoopBarrierCommand;

impl BarrierCommand for NoopBarrierCommand {
    type Error = ();

    #[inline]
    fn run(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// A `CyclicBarrier` is a synchronization aid that allows a set of threads to all wait for
/// each other to reach a *common barrier point*. `CyclicBarrier`s are useful in programs involving
/// a fixed sized party of threads that must occasionally wait for each other. The barrier is called
/// **cyclic** because it can be re-used after the waiting threads are released.
///
/// The port of Java's `CyclicBarrier` converges with that of Rust's `std::sync::Barrier` with some
/// differences.
///
/// A `CyclicBarrier` can be constructed with an optional command that is run once per barrier
/// point, after the *last* thread in the party arrives, but before any threads are released. This
/// **barrier action** is useful for updating any shared state before any of the parties continue.
///
/// If the barrier action does not rely on the parties being suspended when it is executed, then any
/// of the threads in the party could execute that action when it is released. To facilitate this,
/// each invocation of `CyclicBarrier::wait` returns the arrival index of that thread at the
/// barrier. You can then choose which thread should execute the barrier action, for example:
///
/// ```ignore
/// match barrier.wait() {
///     Ok(n) => {
///         // log the completion of this iteration
///     }
///     // ...
/// }
/// ```
///
/// The `CyclicBarrier` uses an all-or-none breakage model for failed synchronization attempts: If a
/// thread leaves a barrier point prematurely because of interruption, failure, or timeout, all
/// other threads waiting at that barrier point will also leave abnormally via
/// `BarrierStatus::Broken` or `BarrierStatus::Interrupted` if they too were interrupted at about
/// the same time.
///
/// # Memory Consistency Effects
///
/// Actions in a thread prior to calling `wait` *happen-before* actions that are part of the
/// barrier action, which in turn *happen-before* actions following a successful return from the
/// corresponding `wait` in other threads.
#[derive(Debug)]
pub struct CyclicBarrier<F>
where
    F: BarrierCommand,
{
    /// We use a `ReentrantMutex` for guarding barrier entry – we need a `reentrant` mutex so a
    /// single thread can repeatly try to acquire the same lock (even if it already has the lock)
    /// so we don't deadlock.
    ///
    /// We need `RefCell` here since we do need interior mutability.
    lock: ReentrantMutex<RefCell<BarrierState>>,
    /// We use a condition variable to wait on until the barrier is tripped.
    trip: ReentrantMutexCondvar,
    /// Number of parties.
    parties: usize,
    /// An *optional* command to run when the barrier is tripped.
    ///
    /// We do *not* need ownership of this command.
    barrier_command: Option<F>,
}

/// We need a custom `Condvar` implemenation that is suitable for use with a `ReentrantMutex` which
/// gives us a `ReentrantMutexGuard` for scoped-based/RAII unlocking. Simple wrapper type.
///
/// See issue [parking_lot#165](https://github.com/Amanieu/parking_lot/issues/165).
#[derive(Debug)]
struct ReentrantMutexCondvar {
    c: Condvar,
    m: Mutex<()>,
}

impl ReentrantMutexCondvar {
    #[inline]
    fn new() -> Self {
        Self {
            c: Condvar::new(),
            m: Mutex::new(()),
        }
    }

    #[inline]
    fn wait<T>(&self, g: &mut ReentrantMutexGuard<'_, T>) {
        let guard = self.m.lock();
        ReentrantMutexGuard::unlocked(g, || {
            // Move the guard in so it gets unlocked before we re-lock g
            let mut guard = guard;
            self.c.wait(&mut guard);
        });
    }

    #[inline]
    fn wait_for<T>(&self, g: &mut ReentrantMutexGuard<'_, T>, timeout: Duration) -> bool {
        let guard = self.m.lock();
        ReentrantMutexGuard::unlocked(g, || {
            // Move the guard in so it gets unlocked before we re-lock g
            let mut guard = guard;
            self.c.wait_for(&mut guard, timeout).timed_out()
        })
    }

    #[inline]
    fn notify_all(&self) {
        self.c.notify_all();
    }
}

/// The inner barrier state. We need to maintain a generation count as well as track how many
/// parties remain. We need the guard this shared mutable state with a mutex so we don't have
/// data races between contending threads. Read/write contention is anticipated to be low, subject
/// to the number of threads that need to rendezvous at the barrier point – the more number of
/// threads, the higher the contention.
#[derive(Debug)]
struct BarrierState {
    /// The current generation.
    generation: Generation,
    /// Number of parties still waiting. Counts down from parties to 0 on each generation. It is
    /// reset to `parties` on each new generation or when broken.
    count: usize,
}

impl<F> CyclicBarrier<F>
where
    F: BarrierCommand,
{
    /// Construct a new `CyclicBarrier` with no barrier command to run when the barrier is tripped.
    ///
    /// # Arguments
    ///
    /// * `parties_count`: number of threads that must invoke `wait` before the barrier is
    ///   tripped. Must be `>= 1`.
    ///
    /// # Panics
    ///
    /// This method panics if `parties < 1`.
    ///
    /// # Implementation Notes
    ///
    /// Right now we're stuck on waiting for `specialization` / `higher-kinded types` to land
    /// before we can make `CyclicBarrier::new()` more ergonomic by not requiring the user
    /// to specify the exact type of `BarrierCommand` since they don't use it. We could specify
    /// that the type is `NoopBarrierCommand` and be done with it.
    #[inline]
    pub fn new(parties: usize) -> Self {
        assert!(parties >= 1);

        Self {
            lock: ReentrantMutex::new(RefCell::new(BarrierState {
                generation: Generation::new(),
                count: parties,
            })),
            trip: ReentrantMutexCondvar::new(),
            parties,
            barrier_command: None,
        }
    }

    /// Construct a new `CyclicBarrier` with a `barrier_command` that is run when the barrier gets
    /// tripped.
    ///
    /// # Arguments
    ///
    /// * `parties_count`: number of threads that must invoke `wait` before the barrier is
    ///   tripped. Must be `>= 1`.
    /// * `barrier_command`: command to execute when the barrier is tripped.
    ///
    /// # Panics
    ///
    /// This method panics if `parties < 1`.
    #[inline]
    pub fn with_barrier_command(parties: usize, barrier_command: F) -> Self {
        assert!(parties >= 1);

        Self {
            lock: ReentrantMutex::new(RefCell::new(BarrierState {
                generation: Generation::new(),
                count: parties,
            })),
            trip: ReentrantMutexCondvar::new(),
            parties,
            barrier_command: Some(barrier_command),
        }
    }

    /// Get the number of parties required to trip the `CyclicBarrier`.
    #[inline]
    pub fn parties(&self) -> usize {
        self.parties
    }

    /// Waits until *all parties* have invoked `wait` on this barrier.
    ///
    /// If the current thread is not the last thread to arrive then it is disabled for thread
    /// scheduling and lies dormant until one of the following happens:
    ///
    /// - The last thread arrives; or
    /// - Some other thread times out while waiting for this barrier; or
    /// - Some other thread invokes `reset` on this barrier.
    ///
    /// If the barrier is `reset` while any thread is waiting, or if the barrier `is_broken` when
    /// `wait` is invoked, or while any thread is waiting, then an `BarrierError::Broken` is
    /// returned.
    ///
    /// If the current thread is the last thread to arrive, and a barrier action was specified,
    /// then the current thread runs the action before allowing other threads to continue.
    ///
    /// If an error `E` is encountered when executing the barrier action, then this error is
    /// propagated to the caller via `BarrerError::BarrierCommandError(E)` and the barrier is placed
    /// in the broken state.
    ///
    /// # Note
    ///
    /// This method is called `CyclicBarrier#await()` in the Java implementation, but in Rust
    /// `await` is a keyword so we'll have to use a different name such as `wait`.
    ///
    /// # Return Values
    ///
    /// * `Ok(usize)`: *arrival index* of the current thread, where index `parties() - 1` indicates
    ///   the first to arrive and zero indicates the last to arrive.
    /// * `Err(BarrierError<E>)`: if timeout expired since construction or reset, or barrier broken,
    ///   or if barrier action failed with error cause `E`.
    #[inline]
    pub fn wait(&self) -> Result<usize, BarrierError<F::Error>> {
        self.do_wait(None)
    }

    /// Waits until *all parties* have invoked `wait` on this barrier, or the specified `timeout`
    /// elapses.
    ///
    /// If the current thread is not the last to arrive then it is disabled for thread scheduling
    /// and lies dormant until one of the following things happens:
    ///
    /// - The last thread arrives; or
    /// - The specified timeout elapses; or
    /// - Some other thread times out while waiting for barrier; or
    /// - Some other thread invokes `reset` on this barrier.
    ///
    /// If the specified waiting time elapses then `BarrierError::TimedOut` is returned. If the
    /// timeout is zero, the method will not wait at all.
    ///
    /// If the barrier is `reset` while any thread is waiting, or if the barrier `is_broken` when
    /// `wait` is invoked, or while any thread is waiting, then `BarrierError::Broken` is returned.
    ///
    /// If the current thread is the last thread to arrive, and a `barrier_command` was specified,
    /// then the current thread runs the action before allowing other threads to continue.
    ///
    /// If any error `E` occurs in the barrier command then that error is propagated to the caller
    /// from the current thread via `BarrierError::BarrierCommandError(E)` and the barrier is placed
    /// in the broken state.
    ///
    /// # Parameters
    ///
    /// * `timeout` - Amount of time to wait before the barrier is tripped.
    ///
    /// # Return Values
    ///
    /// * `Ok(usize)`: *arrival index* of the current thread, where index `parties() - 1` indicates
    ///   the first to arrive and zero indicates the last to arrive.
    /// * `Err(BarrierError<E>)`: if timeout expired since construction or reset, or barrier broken,
    ///   or if barrier action failed with error cause `E`.
    #[inline]
    pub fn wait_for(&self, timeout: Duration) -> Result<usize, BarrierError<F::Error>> {
        self.do_wait(Some(timeout))
    }

    /// Queries if this barrier is in a broken state.
    ///
    /// # Return Value
    ///
    /// Returns `true` if one or more parties broke out of this barrier due to timeout since
    /// construction or last reset, or a barrier action failed due to an error `E`. Returns `false`
    /// otherwise.
    #[inline]
    pub fn is_broken(&self) -> bool {
        let guard = self.lock.lock();
        let state = guard.borrow();
        state.generation.broken()
    }

    /// Resets the barrier to its initial state.  If any parties are currently waiting at the
    /// barrier, they will return with a `BarrierError::Broken`. Resets *after* a breakage has
    /// occurred for other reasons can be complicated to  carry out; threads need to re-synchronize
    /// in some other way, and choose one to perform the reset.  It may be preferable to instead
    /// create a new barrier for subsequent use.
    #[inline]
    pub fn reset(&self) {
        let mut guard = self.lock.lock();

        // Break current generation.
        self.break_barrier(&mut guard);
        // Start new generation.
        self.next_generation(&mut guard);
    }

    /// Returns the number of parties currently blocked at the barrier waiting.
    ///
    /// Primarily useful for debugging and assertions.
    #[inline]
    pub fn parties_waiting(&self) -> usize {
        let guard = self.lock.lock();
        let state = guard.borrow();

        self.parties - state.count
    }

    /// Sets current barrier generation as broken and wakes up everyone.
    #[inline]
    fn break_barrier(&self, guard: &mut ReentrantMutexGuard<'_, RefCell<BarrierState>>) {
        let mut state = guard.borrow_mut();
        state.generation.break_gen();
        state.count = self.parties;

        self.trip.notify_all();
    }

    /// Updates state on barrier trip and wakes up everyone.
    #[inline]
    fn next_generation(&self, guard: &mut ReentrantMutexGuard<'_, RefCell<BarrierState>>) {
        // Signal completion of last generation.
        self.trip.notify_all();

        let mut state = guard.borrow_mut();

        state.count = self.parties;
        let old_gen_id = state.generation.id;
        let new_gen_id = old_gen_id.wrapping_add(1);
        state.generation = Generation::with_id(new_gen_id);
    }

    /// Main barrier code. Note that Rust uses native OS threads and doesn't have the notion of
    /// some "interrupt flag", which is why we don't check if the current thread or any other
    /// thread is interrupted.
    fn do_wait(&self, timeout: Option<Duration>) -> Result<usize, BarrierError<F::Error>> {
        let mut guard = self.lock.lock();
        let g = guard.borrow().generation.clone();

        if g.broken() {
            return Err(BarrierError::Broken);
        }

        let index = {
            let mut state = guard.borrow_mut();
            state.count -= 1;
            state.count
        };

        // If we are the last thread, then we tripped the barrier!
        if index == 0 {
            if let Some(ref cmd) = self.barrier_command {
                if let Err(e) = cmd.run() {
                    return Err(BarrierError::BarrierCommandError(e));
                }
            }
            self.next_generation(&mut guard);
            return Ok(0);
        }

        loop {
            let mut timed_out = false;

            match timeout {
                Some(t) if t.as_nanos() > 0 => {
                    timed_out = self.trip.wait_for(&mut guard, t);
                }
                _ => {
                    self.trip.wait(&mut guard);
                }
            }

            {
                let state = guard.borrow();
                if g.broken() {
                    return Err(BarrierError::Broken);
                }

                if g != state.generation {
                    return Ok(index);
                }
            }

            if timeout.is_some() && timed_out {
                self.break_barrier(&mut guard);
                return Err(BarrierError::TimedOut);
            }
        }
    }
}

/// Errors encountered when using the `CyclicBarrier`.
#[derive(Debug, Eq, PartialEq, Copy, Clone, Ord, PartialOrd)]
pub enum BarrierError<E> {
    /// We timed out while waiting for the threads to reach the common barrier point.
    TimedOut,
    /// The barrier has been broken, e.g. due to errors from underlying threads.
    Broken,
    /// The barrier command failed with error cause `E`.
    BarrierCommandError(E),
}

/// Each use of the barrier is represented as a `Generation` instance. The generation changes
/// whenever the barrier is tripped, or is reset. There can be many generations associated with
/// threads using the barrier - due to the non-deterministic way the lock may be allocated to
/// waiting threads - but only one of these can be active at a time (the one to which `count`
/// applies) and all the rest are either broken or tripped. There need not be an active generation
/// if there has been a break but no subsequent reset.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
struct Generation {
    /// Is the barrier broken?
    broken: bool,
    /// Generation counter. Should be monotonically increasing (subject to wrap-around).
    ///
    /// We need this additional field because the Java implementation can reply on the assumption
    /// that allocating a new object is guaranteed to have a distinct memory address, so their
    /// `==` compares two object references to see if they refer to the same instance. We can't
    /// do that here because we don't have the guarantee that two allocations necessarily have
    /// distinct memory locations (subject to the system allocator).
    id: usize,
}

impl Generation {
    /// Construct a new `Generation`.
    #[inline]
    fn new() -> Self {
        // Initially the barrier is *not* broken.
        Self {
            broken: false,
            id: 0,
        }
    }

    /// Construct a new `Generation` with given generation `id`.
    #[inline]
    fn with_id(id: usize) -> Self {
        // Initially the barrier is *not* broken.
        Self { broken: false, id }
    }

    /// Break the barrier.
    #[inline]
    fn break_gen(&mut self) {
        self.broken = true;
    }

    /// Query barrier integrity.
    #[inline]
    fn broken(&self) -> bool {
        self.broken
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::{CyclicBarrier, NoopBarrierCommand};

    #[test]
    fn test_barrier() {
        const THREADS_COUNT: usize = 10;

        let mut handles = Vec::new();
        let barrier = Arc::new(CyclicBarrier::<NoopBarrierCommand>::new(5));

        for i in 0..THREADS_COUNT {
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                println!("this is thread number {}", i);

                thread::sleep(Duration::from_millis((i * 1000) as u64));

                
                println!("thread number {} hitting barrier", i);

                barrier.wait();

                println!("thread number {} released from barrier", i);

                thread::sleep(Duration::from_secs(5));
            }))
        }

        for handle in handles {
            let _ = handle.join();
        }
    }
}
