# Create .env file if it doesn't exist
$(shell [ ! -f .env ] && touch .env)

# Load environment variables from .env file
include .env

# Set default env variables if not set in .env
export RUST_BACKTRACE ?= full
export RUST_LOG ?= info,nockchain=info,nockchain_libp2p_io=info,libp2p=info,libp2p_quic=info
export MINIMAL_LOG_FORMAT ?= true
export MINING_PKH ?= 9yPePjfWAdUnzaQKyxcRXKRa5PpUzKKEwtpECBZsUYt9Jd7egSDEWoV
export

.PHONY: build
build: build-hoon-all build-rust
	$(call show_env_vars)

## Build all rust
.PHONY: build-rust
build-rust:
	cargo build --release

.PHONY: build-nockchain-jemalloc
build-nockchain-jemalloc:
	cargo build --release --features jemalloc --bin nockchain

## Run all tests
.PHONY: test
test:
	cargo test --release

.PHONY: sol-guard-verify
sol-guard-verify:
	./scripts/verify_sol_guard_plan.sh
	cargo test -p nockchain-bench sol_guard

.PHONY: sol-guard-ci
sol-guard-ci:
	./scripts/sol_guard_ci.sh --help

.PHONY: scope-contract-verify
scope-contract-verify:
	./scripts/verify_scope_evidence_contract.sh
	@[ -f checkpoints/scope_evidence_contract_implementation.md ] || (echo "Missing checklist: checkpoints/scope_evidence_contract_implementation.md" >&2; exit 1)
	@for id in S006 S007 S008 S009 S010; do \
		rg -q "^- \\[[ xX]\\] $$id\\b" checkpoints/scope_evidence_contract_implementation.md || (echo "Missing required checklist ID: $$id" >&2; exit 1); \
	done
	@unchecked="$$(rg -n "^- \\[ \\] S00(6|7|8|9|10)\\b" checkpoints/scope_evidence_contract_implementation.md || true)"; \
	if [ -n "$$unchecked" ]; then \
		echo "Required checklist IDs are unchecked:" >&2; \
		echo "$$unchecked" >&2; \
		exit 1; \
	fi

.PHONY: master-compat-verify
master-compat-verify:
	./scripts/verify_master_compat_inventory.sh
	@[ -f checkpoints/master_compat_inventory_implementation.md ] || (echo "Missing checklist: checkpoints/master_compat_inventory_implementation.md" >&2; exit 1)
	@for id in M006 M007 M008 M009 M010; do \
		rg -q "^- \\[[ xX]\\] $$id\\b" checkpoints/master_compat_inventory_implementation.md || (echo "Missing required checklist ID: $$id" >&2; exit 1); \
	done
	@unchecked="$$(rg -n "^- \\[ \\] M00(6|7|8|9|10)\\b" checkpoints/master_compat_inventory_implementation.md || true)"; \
	if [ -n "$$unchecked" ]; then \
		echo "Required checklist IDs are unchecked:" >&2; \
		echo "$$unchecked" >&2; \
		exit 1; \
	fi

.PHONY: provenance-timeline-verify
provenance-timeline-verify:
	./scripts/verify_provenance_timeline.sh
	@[ -f checkpoints/provenance_timeline_implementation.md ] || (echo "Missing checklist: checkpoints/provenance_timeline_implementation.md" >&2; exit 1)
	@for id in P006 P007 P008 P009 P010; do \
		rg -q "^- \\[[ xX]\\] $$id\\b" checkpoints/provenance_timeline_implementation.md || (echo "Missing required checklist ID: $$id" >&2; exit 1); \
	done
	@unchecked="$$(rg -n "^- \\[ \\] P00(6|7|8|9|10)\\b" checkpoints/provenance_timeline_implementation.md || true)"; \
	if [ -n "$$unchecked" ]; then \
		echo "Required checklist IDs are unchecked:" >&2; \
		echo "$$unchecked" >&2; \
		exit 1; \
	fi

.PHONY: fmt
fmt:
	cargo fmt

.PHONY: build-hoonc
build-hoonc: nuke-hoonc-data ## Build hoonc from this repo
	$(call show_env_vars)
	cargo build --release --locked --bin hoonc

.PHONY: build-hoonc-tracing
build-hoonc-tracing: nuke-hoonc-data ## Build hoonc with tracing
	$(call show_env_vars)
	cargo build --release --bin hoonc --features tracing-tracy

.PHONY: install-hoonc
install-hoonc: nuke-hoonc-data ## Install hoonc from this repo
	$(call show_env_vars)
	cargo install --locked --force --path crates/hoonc --bin hoonc

.PHONY: update-hoonc
update-hoonc:
	$(call show_env_vars)
	cargo install --locked --path crates/hoonc --bin hoonc

.PHONY: build-nockchain
build-nockchain: assets/dumb.jam assets/miner.jam
	$(call show_env_vars)
	cargo build --release --bin nockchain --features tracing-tracy

.PHONY: install-nockchain
install-nockchain: assets/dumb.jam assets/miner.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain --bin nockchain

.PHONY: install-nockchain-wallet
install-nockchain-wallet: assets/wal.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain-wallet --bin nockchain-wallet

.PHONY: install-nockchain-peek
install-nockchain-peek: assets/peek.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain-peek --bin nockchain-peek

.PHONY: ensure-dirs
ensure-dirs:
	mkdir -p hoon
	mkdir -p assets

.PHONY: build-trivial
build-trivial: ensure-dirs
	$(call show_env_vars)
	echo '%trivial' > hoon/trivial.hoon
	hoonc --arbitrary hoon/trivial.hoon

HOON_TARGETS=assets/dumb.jam assets/wal.jam assets/miner.jam assets/peek.jam assets/bridge.jam

.PHONY: nuke-hoonc-data
nuke-hoonc-data:
	rm -rf .data.hoonc
	rm -rf ~/.nockapp/hoonc

.PHONY: nuke-assets
nuke-assets:
	rm -f assets/*.jam

.PHONY: build-hoon-all
build-hoon-all: nuke-assets update-hoonc ensure-dirs build-trivial $(HOON_TARGETS)
	$(call show_env_vars)

.PHONY: build-hoon
build-hoon: ensure-dirs update-hoonc $(HOON_TARGETS)
	$(call show_env_vars)

.PHONY: build-assets
build-assets: ensure-dirs $(HOON_TARGETS)
	$(call show_env_vars)

HOON_SRCS := $(find hoon -type file -name '*.hoon')

## Build dumb.jam with hoonc
assets/dumb.jam: ensure-dirs hoon/apps/dumbnet/outer.hoon $(HOON_SRCS)
	$(call show_env_vars)
	rm -f assets/dumb.jam
	hoonc hoon/apps/dumbnet/outer.hoon hoon
	mv out.jam assets/dumb.jam

## Build wal.jam with hoonc
assets/wal.jam: ensure-dirs hoon/apps/wallet/wallet.hoon $(HOON_SRCS)
	$(call show_env_vars)
	rm -f assets/wal.jam
	hoonc hoon/apps/wallet/wallet.hoon hoon
	mv out.jam assets/wal.jam

## Build mining.jam with hoonc
assets/miner.jam: ensure-dirs hoon/apps/dumbnet/miner.hoon $(HOON_SRCS)
	$(call show_env_vars)
	rm -f assets/miner.jam
	hoonc hoon/apps/dumbnet/miner.hoon hoon
	mv out.jam assets/miner.jam

## Build peek.jam with hoonc
assets/peek.jam: ensure-dirs hoon/apps/peek/peek.hoon $(HOON_SRCS)
	$(call show_env_vars)
	rm -f assets/peek.jam
	hoonc hoon/apps/peek/peek.hoon hoon
	mv out.jam assets/peek.jam

## Build bridge.jam
assets/bridge.jam: ensure-dirs hoon/apps/bridge/bridge.hoon $(HOON_SRCS)
	$(call show_env_vars)
	rm -f assets/bridge.jam
	hoonc hoon/apps/bridge/bridge.hoon hoon
	mv out.jam assets/bridge.jam
