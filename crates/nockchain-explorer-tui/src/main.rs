use std::collections::BTreeMap;
use std::io;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use arboard::Clipboard;
use chrono::{Duration as ChronoDuration, TimeZone, Utc};
use clap::Parser;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use nockapp_grpc_proto::pb::common::v1::{self as pb_common, Base58Hash, PageRequest};
use nockapp_grpc_proto::pb::public::v2::nockchain_block_service_client::NockchainBlockServiceClient;
use nockapp_grpc_proto::pb::public::v2::{
    get_blocks_response, get_transaction_block_response, get_transaction_details_response,
    BlockEntry, GetBlocksRequest, GetTransactionBlockRequest, GetTransactionDetailsRequest,
    TransactionBlockData, TransactionDetails as RpcTransactionDetails,
};
use nockchain_math::belt::Belt;
use nockchain_types::tx_engine::common::Hash as Tip5Hash;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use tonic::Request;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "nockchain-explorer-tui")]
#[command(about = "Block Explorer TUI for Nockchain", long_about = None)]
struct Args {
    /// gRPC server URI (e.g., http://localhost:50051)
    #[arg(short, long, default_value = "http://localhost:50051")]
    server: String,

    /// Fail immediately if cannot connect to server (old behavior)
    #[arg(long)]
    fail_fast: bool,
}

#[derive(Debug, Clone)]
enum View {
    BlocksList,
    BlockDetails(usize), // index in blocks list
    TransactionDetails { block_idx: usize, tx_idx: usize },
    TransactionSearch,
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ConnectionStatus {
    NeverConnected,
    Connected,
    Disconnected,
    Reconnecting,
}

const PAGE_JUMP: usize = 20;
const AUTO_REFRESH_IDLE_GRACE: Duration = Duration::from_secs(3);
const EMPTY_CACHE_BACKOFF: Duration = Duration::from_secs(30);
const ERROR_REFRESH_BACKOFF: Duration = Duration::from_secs(5);
const NICKS_PER_NOCK: u64 = 65_536;

struct App {
    client: Option<NockchainBlockServiceClient<tonic::transport::Channel>>,
    blocks: Vec<BlockEntry>,
    cached_blocks: BTreeMap<u64, BlockEntry>,
    current_height: u64,
    list_state: ListState,
    view: View,
    next_page_token: Option<String>,
    has_more_pages: bool,
    loading: bool,
    error_message: Option<String>,
    status_message: Option<String>,
    clear_status_on_input: bool,
    last_refresh: Instant,
    next_allowed_refresh: Instant,
    auto_refresh_enabled: bool,
    tx_search_input: String,
    tx_search_result: Option<TxSearchResult>,
    server_uri: String,
    previous_view: Option<View>,
    tx_list_state: ListState,
    tx_detail: Option<TxDetailState>,
    clipboard: Option<Clipboard>,
    block_focus: BlockDetailsFocus,
    last_user_action: Instant,

    // Connection state
    connection_status: ConnectionStatus,
    last_successful_connection: Option<Instant>,
    last_connection_attempt: Instant,
    last_connection_error: Option<String>,
    fail_fast: bool,
}

#[derive(Debug, Clone)]
enum TxSearchResult {
    Found(TransactionBlockData),
    Pending,
    NotFound,
    Error(String),
}

#[derive(Debug, Clone)]
enum TxDetailStatus {
    Confirmed(RpcTransactionDetails),
    Pending,
    NotFound,
    Error(String),
}

#[derive(Debug, Clone)]
struct TxDetailState {
    tx_id: String,
    status: TxDetailStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockDetailsFocus {
    Block,
    Transactions,
}

impl App {
    async fn new(server_uri: String, fail_fast: bool) -> Result<Self> {
        // Try to connect, but don't fail if we can't (unless fail_fast is set)
        let (client, connection_status, connection_error) =
            match NockchainBlockServiceClient::connect(server_uri.clone()).await {
                Ok(client) => (Some(client), ConnectionStatus::Connected, None),
                Err(e) => {
                    if fail_fast {
                        return Err(anyhow!("Failed to connect to gRPC server").context(e));
                    }
                    warn!("Initial connection failed: {}, will retry in background", e);
                    (None, ConnectionStatus::NeverConnected, Some(e.to_string()))
                }
            };

        let mut app = Self {
            client,
            blocks: Vec::new(),
            cached_blocks: BTreeMap::new(),
            current_height: 0,
            list_state: ListState::default(),
            view: View::BlocksList,
            next_page_token: None,
            has_more_pages: true,
            loading: false,
            error_message: connection_error.clone(),
            status_message: None,
            clear_status_on_input: false,
            last_refresh: Instant::now(),
            next_allowed_refresh: Instant::now(),
            auto_refresh_enabled: true,
            tx_search_input: String::new(),
            tx_search_result: None,
            server_uri,
            previous_view: None,
            tx_list_state: ListState::default(),
            tx_detail: None,
            clipboard: Clipboard::new().ok(),
            block_focus: BlockDetailsFocus::Block,
            last_user_action: Instant::now(),

            // Connection state
            connection_status,
            last_successful_connection: if connection_status == ConnectionStatus::Connected {
                Some(Instant::now())
            } else {
                None
            },
            last_connection_attempt: Instant::now(),
            last_connection_error: connection_error,
            fail_fast,
        };

        // Only try to load blocks if connected
        if app.connection_status == ConnectionStatus::Connected {
            let _ = app.load_blocks(None).await; // Don't fail if this errors
        }

        Ok(app)
    }

    async fn load_blocks(&mut self, page_token: Option<String>) -> Result<()> {
        let Some(ref mut client) = self.client else {
            self.error_message = Some("Not connected to server".into());
            self.defer_auto_refresh(ERROR_REFRESH_BACKOFF);
            return Ok(());
        };

        self.loading = true;
        self.error_message = None;

        let request = GetBlocksRequest {
            page: Some(PageRequest {
                page_token: page_token.clone().unwrap_or_default(),
                client_page_items_limit: 50,
                max_bytes: 0,
            }),
        };

        match client.get_blocks(Request::new(request)).await {
            Ok(response) => {
                // Mark as connected on successful response
                if self.connection_status != ConnectionStatus::Connected {
                    self.connection_status = ConnectionStatus::Connected;
                    self.last_successful_connection = Some(Instant::now());
                }

                let resp = response.into_inner();
                match resp.result {
                    Some(get_blocks_response::Result::Blocks(blocks_data)) => {
                        if !blocks_data.blocks.is_empty() {
                            self.integrate_blocks(blocks_data.blocks);
                        } else if page_token.is_some() {
                            self.has_more_pages = false;
                            self.next_page_token = None;
                        } else if self.blocks.is_empty() {
                            self.cached_blocks.clear();
                            self.rebuild_blocks(None);
                        }

                        self.current_height = blocks_data.current_height;
                        self.record_refresh();
                        if self.current_height == 0 {
                            self.status_message =
                                Some("Waiting for server to sync with the network…".into());
                            self.clear_status_on_input = false;
                            self.defer_auto_refresh(EMPTY_CACHE_BACKOFF);
                        } else if matches!(
                            self.status_message.as_deref(),
                            Some(msg) if msg.contains("Waiting for server to sync")
                        ) {
                            self.status_message = None;
                        }
                    }
                    Some(get_blocks_response::Result::Error(err)) => {
                        self.error_message = Some(format!("API Error: {}", err.message));
                        self.defer_auto_refresh(ERROR_REFRESH_BACKOFF);
                    }
                    None => {
                        self.error_message = Some("Empty response from server".to_string());
                        self.defer_auto_refresh(ERROR_REFRESH_BACKOFF);
                    }
                }
            }
            Err(e) => {
                // Mark as disconnected on error
                self.connection_status = ConnectionStatus::Disconnected;
                self.last_connection_error = Some(e.to_string());
                self.error_message = Some(format!("gRPC Error: {}", e));
                self.defer_auto_refresh(ERROR_REFRESH_BACKOFF);
            }
        }

        self.loading = false;
        Ok(())
    }

    async fn load_next_page(&mut self) -> Result<()> {
        if let Some(token) = self.next_page_token.clone() {
            self.load_blocks(Some(token)).await?;
        }
        Ok(())
    }

    async fn refresh(&mut self) -> Result<()> {
        self.load_blocks(None).await
    }

    async fn attempt_reconnect(&mut self) -> Result<()> {
        self.last_connection_attempt = Instant::now();
        self.connection_status = ConnectionStatus::Reconnecting;

        match NockchainBlockServiceClient::connect(self.server_uri.clone()).await {
            Ok(client) => {
                self.client = Some(client);
                self.connection_status = ConnectionStatus::Connected;
                self.last_successful_connection = Some(Instant::now());
                self.last_connection_error = None;
                self.status_message = Some("Connected to server!".into());
                info!("Reconnected to server at {}", self.server_uri);

                // Try to load initial blocks
                let _ = self.load_blocks(None).await;
                Ok(())
            }
            Err(e) => {
                self.connection_status = if self.last_successful_connection.is_some() {
                    ConnectionStatus::Disconnected
                } else {
                    ConnectionStatus::NeverConnected
                };
                self.last_connection_error = Some(e.to_string());
                Err(anyhow!("Reconnection failed: {}", e))
            }
        }
    }

    fn should_retry_connection(&self) -> bool {
        matches!(
            self.connection_status,
            ConnectionStatus::NeverConnected | ConnectionStatus::Disconnected
        ) && self.last_connection_attempt.elapsed() >= Duration::from_secs(5)
    }

    async fn search_transaction(&mut self, tx_id: &str) -> Result<()> {
        let Some(ref mut client) = self.client else {
            self.tx_search_result = Some(TxSearchResult::Error("Not connected to server".into()));
            return Ok(());
        };

        self.loading = true;
        self.error_message = None;
        self.tx_search_result = None;

        let trimmed = tx_id.trim();
        if trimmed.is_empty() {
            self.tx_search_result = Some(TxSearchResult::Error(
                "Please enter at least one character".into(),
            ));
            self.loading = false;
            return Ok(());
        }

        let request = GetTransactionBlockRequest {
            tx_id: Some(Base58Hash {
                hash: trimmed.to_string(),
            }),
        };

        match client.get_transaction_block(Request::new(request)).await {
            Ok(response) => {
                // Mark as connected on successful response
                if self.connection_status != ConnectionStatus::Connected {
                    self.connection_status = ConnectionStatus::Connected;
                    self.last_successful_connection = Some(Instant::now());
                }

                let resp = response.into_inner();
                self.tx_search_result = Some(Self::map_tx_response(resp.result));
            }
            Err(e) => {
                // Mark as disconnected on error
                self.connection_status = ConnectionStatus::Disconnected;
                self.last_connection_error = Some(e.to_string());
                self.tx_search_result = Some(TxSearchResult::Error(format!("gRPC Error: {}", e)));
            }
        }

        self.loading = false;
        Ok(())
    }

    async fn open_transaction_detail(&mut self, block_idx: usize, tx_idx: usize) -> Result<()> {
        let Some(ref mut client) = self.client else {
            self.error_message = Some("Not connected to server".into());
            return Ok(());
        };

        let tx_id = {
            let block = self
                .blocks
                .get(block_idx)
                .ok_or_else(|| anyhow!("Block index out of range"))?;
            block
                .tx_ids
                .get(tx_idx)
                .ok_or_else(|| anyhow!("Transaction index out of range"))?
                .hash
                .clone()
        };

        self.loading = true;
        self.error_message = None;

        let request = GetTransactionDetailsRequest {
            tx_id: Some(Base58Hash {
                hash: tx_id.clone(),
            }),
        };

        match client.get_transaction_details(Request::new(request)).await {
            Ok(response) => {
                // Mark as connected on successful response
                if self.connection_status != ConnectionStatus::Connected {
                    self.connection_status = ConnectionStatus::Connected;
                    self.last_successful_connection = Some(Instant::now());
                }

                let resp = response.into_inner();
                self.tx_detail = Some(TxDetailState {
                    tx_id: tx_id.clone(),
                    status: Self::map_tx_detail_response(resp.result),
                });
                self.view = View::TransactionDetails { block_idx, tx_idx };
                self.block_focus = BlockDetailsFocus::Transactions;
            }
            Err(e) => {
                // Mark as disconnected on error
                self.connection_status = ConnectionStatus::Disconnected;
                self.last_connection_error = Some(e.to_string());
                self.error_message = Some(format!("gRPC Error: {}", e));
            }
        }

        self.loading = false;
        Ok(())
    }

    fn map_tx_response(result: Option<get_transaction_block_response::Result>) -> TxSearchResult {
        match result {
            Some(get_transaction_block_response::Result::Block(block_data)) => {
                TxSearchResult::Found(block_data)
            }
            Some(get_transaction_block_response::Result::Pending(_)) => TxSearchResult::Pending,
            Some(get_transaction_block_response::Result::Error(err)) => {
                if err.message.contains("not found") {
                    TxSearchResult::NotFound
                } else {
                    TxSearchResult::Error(err.message)
                }
            }
            None => TxSearchResult::Error("Empty response".to_string()),
        }
    }

    fn map_tx_detail_response(
        result: Option<get_transaction_details_response::Result>,
    ) -> TxDetailStatus {
        match result {
            Some(get_transaction_details_response::Result::Details(details)) => {
                TxDetailStatus::Confirmed(details)
            }
            Some(get_transaction_details_response::Result::Pending(_)) => TxDetailStatus::Pending,
            Some(get_transaction_details_response::Result::Error(err)) => {
                if err.message.contains("not found") {
                    TxDetailStatus::NotFound
                } else {
                    TxDetailStatus::Error(err.message)
                }
            }
            None => TxDetailStatus::Error("Empty response".to_string()),
        }
    }

    fn selected_height(&self) -> Option<u64> {
        self.list_state
            .selected()
            .and_then(|idx| self.blocks.get(idx))
            .map(|b| b.height)
    }

    fn integrate_blocks(&mut self, new_blocks: Vec<BlockEntry>) {
        let preferred_height = self.selected_height();
        for block in new_blocks {
            self.cached_blocks.insert(block.height, block);
        }
        self.rebuild_blocks(preferred_height);
    }

    fn rebuild_blocks(&mut self, preferred_height: Option<u64>) {
        self.blocks = self
            .cached_blocks
            .iter()
            .rev()
            .map(|(_, block)| block.clone())
            .collect();

        let target_height = preferred_height.or_else(|| self.blocks.first().map(|b| b.height));
        if let Some(height) = target_height {
            if let Some(idx) = self.blocks.iter().position(|b| b.height == height) {
                self.list_state.select(Some(idx));
            } else if !self.blocks.is_empty() {
                self.list_state.select(Some(0));
            } else {
                self.list_state.select(None);
            }
        } else if !self.blocks.is_empty() {
            self.list_state.select(Some(0));
        } else {
            self.list_state.select(None);
        }

        self.update_pagination_tokens();

        if matches!(
            self.view,
            View::BlockDetails(_) | View::TransactionDetails { .. }
        ) {
            if let Some(idx) = self.list_state.selected() {
                self.view = View::BlockDetails(idx);
            } else {
                self.view = View::BlocksList;
            }
        }
        self.sync_tx_list_selection();
    }

    fn update_pagination_tokens(&mut self) {
        if let Some(oldest) = self.blocks.last() {
            if oldest.height > 0 {
                self.next_page_token = Some(format!("{:x}", oldest.height - 1));
                self.has_more_pages = true;
            } else {
                self.next_page_token = None;
                self.has_more_pages = false;
            }
        } else {
            self.next_page_token = None;
            self.has_more_pages = false;
        }
    }

    fn move_selection_down(&mut self) -> Option<usize> {
        if self.blocks.is_empty() {
            self.list_state.select(None);
            return None;
        }
        let len = self.blocks.len();
        let current = self.list_state.selected().unwrap_or(0);
        let new_idx = if current + 1 < len {
            current + 1
        } else {
            current
        };
        self.list_state.select(Some(new_idx));
        Some(new_idx)
    }

    fn move_selection_up(&mut self) -> Option<usize> {
        if self.blocks.is_empty() {
            self.list_state.select(None);
            return None;
        }
        let current = self.list_state.selected().unwrap_or(0);
        let new_idx = current.saturating_sub(1);
        self.list_state.select(Some(new_idx));
        Some(new_idx)
    }

    fn sync_tx_list_selection(&mut self) {
        self.tx_list_state = ListState::default();
        if let View::BlockDetails(idx) = self.view {
            if let Some(block) = self.blocks.get(idx) {
                if !block.tx_ids.is_empty() {
                    self.tx_list_state.select(Some(0));
                }
            }
        }
    }

    fn viewing_tip(&self) -> bool {
        if self.blocks.is_empty() {
            true
        } else {
            self.list_state
                .selected()
                .map(|idx| idx == 0)
                .unwrap_or(true)
        }
    }

    fn should_auto_refresh_blocks(&self, interval: Duration) -> bool {
        self.auto_refresh_enabled
            && Instant::now() >= self.next_allowed_refresh
            && matches!(self.view, View::BlocksList)
            && self.viewing_tip()
            && self.last_refresh.elapsed() >= interval
            && self.can_auto_refresh(AUTO_REFRESH_IDLE_GRACE)
    }

    fn is_at_bottom(&self) -> bool {
        match self.list_state.selected() {
            Some(idx) if !self.blocks.is_empty() => idx == self.blocks.len() - 1,
            _ => false,
        }
    }

    fn should_auto_fetch_more(&self) -> bool {
        self.has_more_pages && !self.loading && self.is_at_bottom()
    }

    fn page_down(&mut self) -> Option<usize> {
        if self.blocks.is_empty() {
            self.list_state.select(None);
            return None;
        }
        let len = self.blocks.len();
        let current = self.list_state.selected().unwrap_or(0);
        let jump = PAGE_JUMP.min(len.saturating_sub(1));
        let new_idx = (current + jump).min(len - 1);
        self.list_state.select(Some(new_idx));
        Some(new_idx)
    }

    fn page_up(&mut self) -> Option<usize> {
        if self.blocks.is_empty() {
            self.list_state.select(None);
            return None;
        }
        let current = self.list_state.selected().unwrap_or(0);
        let jump = PAGE_JUMP.min(current);
        let new_idx = current.saturating_sub(jump);
        self.list_state.select(Some(new_idx));
        Some(new_idx)
    }

    fn current_block(&self) -> Option<&BlockEntry> {
        self.list_state
            .selected()
            .and_then(|idx| self.blocks.get(idx))
    }

    fn selected_tx_index(&self) -> Option<usize> {
        self.tx_list_state.selected()
    }

    fn selected_tx_id(&self) -> Option<String> {
        match (self.current_block(), self.selected_tx_index()) {
            (Some(block), Some(idx)) => block.tx_ids.get(idx).map(|tx| tx.hash.clone()),
            _ => None,
        }
    }

    fn copy_tx_id(&mut self, tx_id: &str) {
        self.copy_text_to_clipboard(tx_id, "Copied transaction id to clipboard");
    }

    fn copy_selected_block_id(&mut self) {
        let block = match self.current_block() {
            Some(block) => block,
            None => {
                self.error_message = Some("No block selected".into());
                self.status_message = None;
                self.clear_status_on_input = false;
                return;
            }
        };

        match hash_option_to_base58(&block.block_id) {
            Some(id) => self.copy_text_to_clipboard(&id, "Copied block id to clipboard"),
            None => {
                self.error_message = Some("Block id unavailable".into());
                self.status_message = None;
                self.clear_status_on_input = false;
            }
        }
    }

    fn copy_text_to_clipboard(&mut self, value: &str, success_message: &str) {
        self.error_message = None;
        match self.clipboard.as_mut() {
            Some(clip) => {
                if let Err(e) = clip.set_text(value.to_string()) {
                    self.error_message = Some(format!("Failed to copy: {}", e));
                    self.status_message = None;
                    self.clear_status_on_input = false;
                } else {
                    self.status_message = Some(success_message.into());
                    self.clear_status_on_input = true;
                    self.error_message = None;
                }
            }
            None => match Clipboard::new() {
                Ok(mut clip) => {
                    if let Err(e) = clip.set_text(value.to_string()) {
                        self.error_message = Some(format!("Failed to copy: {}", e));
                        self.status_message = None;
                        self.clear_status_on_input = false;
                    } else {
                        self.status_message = Some(success_message.into());
                        self.clear_status_on_input = true;
                        self.error_message = None;
                        self.clipboard = Some(clip);
                    }
                }
                Err(e) => {
                    self.error_message = Some(format!("Clipboard unavailable: {}", e));
                    self.status_message = None;
                    self.clear_status_on_input = false;
                }
            },
        }
    }

    fn read_clipboard_text(&mut self) -> Option<String> {
        match self.clipboard.as_mut() {
            Some(clip) => clip.get_text().ok(),
            None => match Clipboard::new() {
                Ok(mut clip) => match clip.get_text() {
                    Ok(text) => {
                        self.clipboard = Some(clip);
                        Some(text)
                    }
                    Err(_) => None,
                },
                Err(_) => None,
            },
        }
    }

    fn clear_status_if_needed(&mut self) {
        if self.clear_status_on_input {
            self.status_message = None;
            self.clear_status_on_input = false;
        }
    }

    fn move_tx_selection(&mut self, delta: i32) {
        let len = match self.current_block() {
            Some(block) => block.tx_ids.len(),
            None => return,
        };
        if len == 0 {
            self.tx_list_state.select(None);
            return;
        }
        let current = self.tx_list_state.selected().unwrap_or(0) as i32;
        let new_idx = (current + delta).clamp(0, (len - 1) as i32) as usize;
        self.tx_list_state.select(Some(new_idx));
    }

    fn note_user_action(&mut self) {
        self.last_user_action = Instant::now();
    }

    fn record_refresh(&mut self) {
        let now = Instant::now();
        self.last_refresh = now;
        self.next_allowed_refresh = now;
    }

    fn defer_auto_refresh(&mut self, delay: Duration) {
        self.next_allowed_refresh = Instant::now() + delay;
    }

    fn can_auto_refresh(&self, interval: Duration) -> bool {
        self.last_user_action.elapsed() >= interval
    }
    fn open_help(&mut self) {
        if !matches!(self.view, View::Help) {
            self.previous_view = Some(self.view.clone());
            self.view = View::Help;
        }
    }

    fn close_help(&mut self) {
        if let Some(prev) = self.previous_view.take() {
            self.view = prev;
        } else {
            self.view = View::BlocksList;
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Main content
            Constraint::Length(3), // Status bar
            Constraint::Length(2), // Help
        ])
        .split(f.area());

    // Header
    render_header(f, chunks[0], app);

    // Main content based on view
    match &app.view {
        View::BlocksList => render_blocks_list(f, chunks[1], app),
        View::BlockDetails(idx) => render_block_details(f, chunks[1], app, *idx),
        View::TransactionDetails { block_idx, tx_idx } => {
            render_transaction_details(f, chunks[1], app, *block_idx, *tx_idx)
        }
        View::TransactionSearch => render_transaction_search(f, chunks[1], app),
        View::Help => render_help_menu(f, chunks[1]),
    }

    // Status bar
    render_status_bar(f, chunks[2], app);

    // Help
    render_help(f, chunks[3], app);
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let connection_indicator = match app.connection_status {
        ConnectionStatus::Connected => {
            let age = app
                .last_successful_connection
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(0);
            Span::styled(
                format!("● Connected ({}s ago)", age),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        }
        ConnectionStatus::Disconnected => Span::styled(
            "● Disconnected",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        ConnectionStatus::Reconnecting => Span::styled(
            "● Reconnecting...",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        ConnectionStatus::NeverConnected => Span::styled(
            "● Never Connected",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
    };

    let title = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                "Nockchain Block Explorer",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            connection_indicator,
            Span::raw(" | "),
            Span::styled(
                format!("Height: {}", app.current_height),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::raw("Server: "),
            Span::styled(&app.server_uri, Style::default().fg(Color::Green)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title("Info"));
    f.render_widget(title, area);
}

fn render_blocks_list(f: &mut Frame, area: Rect, app: &mut App) {
    let items: Vec<ListItem> = app
        .blocks
        .iter()
        .map(|block| {
            let tx_count = block.tx_ids.len();
            let content = vec![Line::from(vec![
                Span::styled(
                    format!("Height: {:6}", block.height),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(" | "),
                Span::styled(
                    format!("TXs: {:3}", tx_count),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" | "),
                Span::styled(
                    format!("Block ID: {}", hash_display(&block.block_id)),
                    Style::default().fg(Color::Green),
                ),
            ])];
            ListItem::new(content)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("Blocks ({})", app.blocks.len())),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    f.render_stateful_widget(list, area, &mut app.list_state);
}

fn render_block_details(f: &mut Frame, area: Rect, app: &mut App, idx: usize) {
    let block = match app.blocks.get(idx) {
        Some(b) => b,
        None => {
            let text = Paragraph::new("Block not found").block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Block Details"),
            );
            f.render_widget(text, area);
            return;
        }
    };

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Height: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(block.height.to_string()),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Block ID: ",
            Style::default().add_modifier(Modifier::BOLD),
        )]),
        Line::from(Span::styled(
            hash_full_display(&block.block_id),
            Style::default().fg(Color::Green),
        )),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Parent ID: ",
            Style::default().add_modifier(Modifier::BOLD),
        )]),
        Line::from(Span::styled(
            hash_full_display(&block.parent),
            Style::default().fg(Color::Yellow),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("Timestamp: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format_timestamp(block.timestamp)),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            format!("Transactions ({}): ", block.tx_ids.len()),
            Style::default().add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
    ];

    if block.tx_ids.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (no transactions)",
            Style::default().fg(Color::DarkGray),
        )));
        let paragraph = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Block Details (ESC to go back)"),
            )
            .wrap(Wrap { trim: false });
        f.render_widget(paragraph, area);
        return;
    }

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(lines.len() as u16), Constraint::Min(5)])
        .split(area);

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Block Details (ESC to go back)"),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, layout[0]);

    let tx_items: Vec<ListItem> = block
        .tx_ids
        .iter()
        .enumerate()
        .map(|(i, tx)| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("[{:3}] ", i + 1),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(&tx.hash, Style::default().fg(Color::Cyan)),
            ]))
        })
        .collect();

    let title = match app.block_focus {
        BlockDetailsFocus::Transactions => "Transactions (Enter to inspect, c copies tx id)",
        BlockDetailsFocus::Block => "Transactions (Tab to focus, Enter opens details)",
    };

    let list = List::new(tx_items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");
    f.render_stateful_widget(list, layout[1], &mut app.tx_list_state);
}

fn render_transaction_details(
    f: &mut Frame,
    area: Rect,
    app: &App,
    block_idx: usize,
    tx_idx: usize,
) {
    let tx_state = match &app.tx_detail {
        Some(detail) => detail,
        None => {
            let paragraph = Paragraph::new("Transaction details not loaded")
                .block(Block::default().borders(Borders::ALL).title("Transaction"));
            f.render_widget(paragraph, area);
            return;
        }
    };

    match &tx_state.status {
        TxDetailStatus::Confirmed(details) => {
            let header_lines = build_tx_header_lines(tx_state, details, block_idx, tx_idx);
            let header_height = header_lines.len().saturating_add(2) as u16;
            let header_constraint =
                Constraint::Length(header_height.min(area.height.saturating_sub(6).max(6)));

            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([header_constraint, Constraint::Min(0)])
                .split(area);

            let header = Paragraph::new(header_lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Transaction Details"),
                )
                .wrap(Wrap { trim: false });
            f.render_widget(header, layout[0]);

            let body = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(layout[1]);

            render_tx_inputs_section(f, body[0], details);
            render_tx_outputs_section(f, body[1], details);
        }
        TxDetailStatus::Pending => render_tx_status_message(
            f,
            area,
            tx_state,
            block_idx,
            tx_idx,
            "Pending (not yet included in a block)",
            Color::Yellow,
        ),
        TxDetailStatus::NotFound => render_tx_status_message(
            f,
            area,
            tx_state,
            block_idx,
            tx_idx,
            "Transaction not found on chain or in mempool",
            Color::Red,
        ),
        TxDetailStatus::Error(err) => render_tx_status_message(
            f,
            area,
            tx_state,
            block_idx,
            tx_idx,
            &format!("Error: {}", err),
            Color::Red,
        ),
    }
}

fn build_tx_header_lines(
    tx_state: &TxDetailState,
    details: &RpcTransactionDetails,
    block_idx: usize,
    tx_idx: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            "Transaction ID: ",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(tx_state.tx_id.clone()),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("Confirmed in block {}", details.height),
            Style::default().fg(Color::Green),
        ),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![Span::styled(
        "Block ID: ",
        Style::default().add_modifier(Modifier::BOLD),
    )]));
    lines.push(Line::from(Span::styled(
        hash_full_display(&details.block_id),
        Style::default().fg(Color::Green),
    )));
    lines.push(Line::from(vec![Span::styled(
        "Parent: ",
        Style::default().add_modifier(Modifier::BOLD),
    )]));
    lines.push(Line::from(Span::styled(
        hash_full_display(&details.parent),
        Style::default().fg(Color::Yellow),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("Height: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(details.height.to_string()),
        Span::raw("  Timestamp: "),
        Span::raw(format_timestamp(details.timestamp)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Version: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(details.version.to_string()),
        Span::raw("  Size: "),
        Span::raw(format!("{} bytes", format_number(details.size_bytes))),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Totals: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!(
            "in {} | out {} | fee {}",
            format_amount(details.total_input.as_ref()),
            format_amount(details.total_output.as_ref()),
            format_amount(details.fee.as_ref())
        )),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            "List position: ",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("block {} • tx {}", block_idx, tx_idx)),
    ]));
    lines.push(Line::from(Span::styled(
        "ESC: Back to block • c: Copy tx id",
        Style::default().fg(Color::DarkGray),
    )));
    lines
}

fn render_tx_inputs_section(f: &mut Frame, area: Rect, details: &RpcTransactionDetails) {
    let mut lines = Vec::new();
    if details.inputs.is_empty() {
        lines.push(Line::from("No inputs"));
    } else {
        for (idx, input) in details.inputs.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("[{:02}] ", idx + 1),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{} ", format_amount(input.amount.as_ref())),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled(&input.note_name_b58, Style::default().fg(Color::Cyan)),
            ]));
            let mut source_desc = format!("source: {}", short_hash_str(&input.source_tx_id));
            if input.coinbase {
                source_desc.push_str(" (coinbase)");
            }
            lines.push(Line::from(vec![
                Span::raw("      "),
                Span::styled(source_desc, Style::default().fg(Color::DarkGray)),
            ]));
            lines.push(Line::from(""));
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Inputs"))
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn render_tx_outputs_section(f: &mut Frame, area: Rect, details: &RpcTransactionDetails) {
    let mut lines = Vec::new();
    if details.outputs.is_empty() {
        lines.push(Line::from("No outputs"));
    } else {
        for (idx, output) in details.outputs.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("[{:02}] ", idx + 1),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{} ", format_amount(output.amount.as_ref())),
                    Style::default().fg(Color::LightGreen),
                ),
                Span::styled(&output.note_name_b58, Style::default().fg(Color::Cyan)),
            ]));
            lines.push(Line::from(vec![
                Span::raw("      "),
                Span::styled(
                    format!("lock: {}", output.lock_summary),
                    Style::default().fg(Color::Magenta),
                ),
            ]));
            lines.push(Line::from(""));
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Outputs"))
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn render_tx_status_message(
    f: &mut Frame,
    area: Rect,
    tx_state: &TxDetailState,
    block_idx: usize,
    tx_idx: usize,
    message: &str,
    color: Color,
) {
    let lines = vec![
        Line::from(vec![
            Span::styled(
                "Transaction ID: ",
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(tx_state.tx_id.clone()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(message, Style::default().fg(color)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "List position: ",
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("block {} • tx {}", block_idx, tx_idx)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "ESC: Back to block • c: Copy tx id",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Transaction Details"),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, area);
}

fn format_amount(amount: Option<&pb_common::Nicks>) -> String {
    let value = amount.map(|n| n.value).unwrap_or(0);
    let nock = value as f64 / NICKS_PER_NOCK as f64;
    format!(
        "{} nicks ({})",
        format_number(value),
        format_nock_value(nock)
    )
}

fn format_nock_value(value: f64) -> String {
    let mut s = format!("{:.6}", value);
    while s.contains('.') && s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}

fn short_hash_str(text: &str) -> String {
    if text.len() <= 16 {
        text.to_string()
    } else {
        format!("{}...{}", &text[..8], &text[text.len() - 6..])
    }
}

fn format_number(value: u64) -> String {
    let s = value.to_string();
    let mut acc = String::with_capacity(s.len() + s.len() / 3);
    let mut count = 0;
    for ch in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            acc.push('_');
        }
        acc.push(ch);
        count += 1;
    }
    acc.chars().rev().collect()
}

fn render_transaction_search(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    // Input box
    let input = Paragraph::new(app.tx_search_input.as_str())
        .style(Style::default().fg(Color::Yellow))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Transaction ID prefix (base58, Enter to search)"),
        );
    f.render_widget(input, chunks[0]);

    // Results
    let result_text = match &app.tx_search_result {
        Some(TxSearchResult::Found(block_data)) => {
            vec![
                Line::from(vec![
                    Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(
                        "CONFIRMED",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled(
                        "Block Height: ",
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(block_data.height.to_string()),
                ]),
                Line::from(""),
                Line::from(vec![Span::styled(
                    "Block ID: ",
                    Style::default().add_modifier(Modifier::BOLD),
                )]),
                Line::from(Span::styled(
                    hash_full_display(&block_data.block_id),
                    Style::default().fg(Color::Green),
                )),
                Line::from(""),
                Line::from(vec![Span::styled(
                    "Parent ID: ",
                    Style::default().add_modifier(Modifier::BOLD),
                )]),
                Line::from(Span::styled(
                    hash_full_display(&block_data.parent),
                    Style::default().fg(Color::Yellow),
                )),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Timestamp: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(format_timestamp(block_data.timestamp)),
                ]),
            ]
        }
        Some(TxSearchResult::Pending) => {
            vec![
                Line::from(vec![
                    Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(
                        "PENDING",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(""),
                Line::from("Transaction exists in mempool but not yet in a block."),
            ]
        }
        Some(TxSearchResult::NotFound) => {
            vec![
                Line::from(vec![
                    Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(
                        "NOT FOUND",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(""),
                Line::from("Transaction does not exist in blockchain or mempool."),
            ]
        }
        Some(TxSearchResult::Error(err)) => {
            vec![
                Line::from(vec![Span::styled(
                    "Error: ",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                )]),
                Line::from(""),
                Line::from(Span::raw(err)),
            ]
        }
        None => {
            vec![Line::from(Span::styled(
                "Enter a transaction ID prefix and press Enter (Ctrl+V or Shift+Insert pastes)",
                Style::default().fg(Color::DarkGray),
            ))]
        }
    };

    let results = Paragraph::new(result_text)
        .block(Block::default().borders(Borders::ALL).title("Results"))
        .wrap(Wrap { trim: false });
    f.render_widget(results, chunks[1]);
}

fn render_status_bar(f: &mut Frame, area: Rect, app: &App) {
    let status_text = if let Some(err) = &app.error_message {
        vec![Line::from(vec![
            Span::styled(
                "Error: ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(err, Style::default().fg(Color::Red)),
        ])]
    } else if !matches!(app.connection_status, ConnectionStatus::Connected) {
        // Show connection error when not connected
        let mut lines = vec![Line::from(vec![
            Span::styled(
                "Disconnected: ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                app.last_connection_error
                    .as_deref()
                    .unwrap_or("Unknown error"),
                Style::default().fg(Color::Red),
            ),
        ])];

        if let Some(last_success) = app.last_successful_connection {
            lines.push(Line::from(vec![
                Span::raw("Last successful connection: "),
                Span::styled(
                    format_duration(last_success.elapsed()),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(" ago"),
            ]));
        }

        lines
    } else if app.loading {
        vec![Line::from(Span::styled(
            "Loading...",
            Style::default().fg(Color::Yellow),
        ))]
    } else if let Some(msg) = &app.status_message {
        vec![Line::from(vec![
            Span::styled(
                "Status: ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(msg, Style::default().fg(Color::Cyan)),
        ])]
    } else {
        let age = app.last_refresh.elapsed().as_secs();
        let refresh_status = if app.auto_refresh_enabled {
            if matches!(app.view, View::BlocksList) && app.viewing_tip() {
                format!("Auto-refresh: ON (last: {}s ago)", age)
            } else if !matches!(app.view, View::BlocksList) {
                format!("Auto-refresh: PAUSED (viewing details, last: {}s ago)", age)
            } else {
                format!(
                    "Auto-refresh: PAUSED (scroll to newest block to resume, last: {}s ago)",
                    age
                )
            }
        } else {
            format!("Auto-refresh: OFF (last: {}s ago)", age)
        };
        vec![Line::from(vec![
            Span::styled("Ready", Style::default().fg(Color::Green)),
            Span::raw(" | "),
            Span::raw(refresh_status),
            if app.has_more_pages {
                Span::styled(" | More pages available", Style::default().fg(Color::Cyan))
            } else {
                Span::raw("")
            },
        ])]
    };

    let status =
        Paragraph::new(status_text).block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(status, area);
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

fn render_help(f: &mut Frame, area: Rect, app: &App) {
    let help_text = match &app.view {
        View::BlocksList => {
            "↑: Up | ↓: Down (auto-loads older blocks) | PgUp/PgDn: Jump | Enter: View block | c: Copy block id | t: Search TX | r: Refresh | a: Auto-refresh | n: Next Page | ?: Help | q: Quit"
        }
        View::BlockDetails(_) => {
            "ESC: Back | PgUp/PgDn: Prev/next block | Tab: Toggle tx focus | ↑/↓ (tx focus): Move selection | Enter: TX details | c: Copy tx id (tx focus) | ?: Help | q: Quit"
        }
        View::TransactionDetails { .. } => "ESC: Back to block | c: Copy tx id | ?: Help | q: Quit",
        View::TransactionSearch => {
            "ESC: Back | Enter: Search (prefix ok) | Ctrl+V/Ctrl+Shift+V: Paste | Ctrl+C: Clear | ?: Help | q: Quit"
        }
        View::Help => "ESC/q/?: Close help",
    };

    let help = Paragraph::new(help_text).style(Style::default().fg(Color::DarkGray));
    f.render_widget(help, area);
}

fn hash_display(hash: &Option<nockapp_grpc_proto::pb::common::v1::Hash>) -> String {
    hash.as_ref()
        .map(|h| {
            let full = hash_to_string(h);
            if full.len() > 16 {
                format!("{}...{}", &full[..8], &full[full.len() - 8..])
            } else {
                full
            }
        })
        .unwrap_or_else(|| "(none)".to_string())
}

fn hash_full_display(hash: &Option<nockapp_grpc_proto::pb::common::v1::Hash>) -> String {
    hash.as_ref()
        .map(hash_to_string)
        .unwrap_or_else(|| "(none)".to_string())
}

fn hash_option_to_base58(hash: &Option<pb_common::Hash>) -> Option<String> {
    hash.as_ref()
        .and_then(|h| proto_hash_to_tip5(h))
        .map(|h| h.to_base58())
}

fn proto_hash_to_tip5(hash: &pb_common::Hash) -> Option<Tip5Hash> {
    Some(Tip5Hash([
        Belt(hash.belt_1.as_ref()?.value),
        Belt(hash.belt_2.as_ref()?.value),
        Belt(hash.belt_3.as_ref()?.value),
        Belt(hash.belt_4.as_ref()?.value),
        Belt(hash.belt_5.as_ref()?.value),
    ]))
}

fn hash_to_string(hash: &nockapp_grpc_proto::pb::common::v1::Hash) -> String {
    format!(
        "{:016x}.{:016x}.{:016x}.{:016x}.{:016x}",
        hash.belt_1.as_ref().map(|b| b.value).unwrap_or(0),
        hash.belt_2.as_ref().map(|b| b.value).unwrap_or(0),
        hash.belt_3.as_ref().map(|b| b.value).unwrap_or(0),
        hash.belt_4.as_ref().map(|b| b.value).unwrap_or(0),
        hash.belt_5.as_ref().map(|b| b.value).unwrap_or(0),
    )
}

fn format_timestamp(raw_ts: u64) -> String {
    // Blocks encode timestamps using `time-in-secs` on an Urbit `@da`, which means we need to
    // subtract the Urbit base epoch (2^63 plus an offset) to get back to real Unix seconds.
    const BASE_URBIT_EPOCH: u64 = 0x8000_000c_ce9e_0d80;

    let unix_secs = match raw_ts.checked_sub(BASE_URBIT_EPOCH) {
        Some(secs) => secs as i64,
        None => return format!("{} (before Urbit epoch base)", raw_ts),
    };

    match Utc.timestamp_opt(unix_secs, 0).single() {
        Some(dt) => {
            let age = Utc::now().signed_duration_since(dt);
            format!(
                "{} UTC ({})",
                dt.format("%Y-%m-%d %H:%M:%S"),
                format_relative_duration(age)
            )
        }
        None => format!("{} (invalid timestamp)", raw_ts),
    }
}

fn format_relative_duration(duration: ChronoDuration) -> String {
    let secs = duration.num_seconds();
    if secs == 0 {
        return "just now".to_string();
    }

    let suffix = if secs >= 0 { "ago" } else { "from now" };
    let mut remaining = secs.abs();

    let mut parts = Vec::new();
    let days = remaining / 86_400;
    if days > 0 {
        parts.push(format!("{}d", days));
        remaining %= 86_400;
    }
    let hours = remaining / 3_600;
    if hours > 0 && parts.len() < 2 {
        parts.push(format!("{}h", hours));
        remaining %= 3_600;
    }
    let minutes = remaining / 60;
    if minutes > 0 && parts.len() < 2 {
        parts.push(format!("{}m", minutes));
        remaining %= 60;
    }
    if remaining > 0 && parts.len() < 2 {
        parts.push(format!("{}s", remaining));
    }

    if parts.is_empty() {
        format!("less than 1s {}", suffix)
    } else {
        format!("{} {}", parts.join(" "), suffix)
    }
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(Self { terminal })
    }

    fn terminal(&mut self) -> &mut Terminal<CrosstermBackend<io::Stdout>> {
        &mut self.terminal
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = self.terminal.show_cursor();
    }
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: App,
) -> Result<()> {
    let tick_rate = Duration::from_millis(250);
    let auto_refresh_interval = Duration::from_secs(10);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            match event::read()? {
                Event::Key(key) => {
                    app.clear_status_if_needed();
                    app.note_user_action();
                    match &app.view {
                        View::BlocksList => match key.code {
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Down => {
                                app.move_selection_down();
                                if app.should_auto_fetch_more() {
                                    app.load_next_page().await?;
                                }
                            }
                            KeyCode::Up => {
                                app.move_selection_up();
                            }
                            KeyCode::PageDown => {
                                app.page_down();
                                if app.should_auto_fetch_more() {
                                    app.load_next_page().await?;
                                }
                            }
                            KeyCode::PageUp => {
                                app.page_up();
                            }
                            KeyCode::Char('c') => {
                                app.copy_selected_block_id();
                            }
                            KeyCode::Char('?') => app.open_help(),
                            KeyCode::Enter => {
                                if let Some(idx) = app.list_state.selected() {
                                    app.view = View::BlockDetails(idx);
                                    app.block_focus = if app
                                        .blocks
                                        .get(idx)
                                        .map(|b| b.tx_ids.is_empty())
                                        .unwrap_or(true)
                                    {
                                        BlockDetailsFocus::Block
                                    } else {
                                        BlockDetailsFocus::Transactions
                                    };
                                    app.sync_tx_list_selection();
                                }
                            }
                            KeyCode::Char('t') => {
                                app.view = View::TransactionSearch;
                                app.tx_search_input.clear();
                                app.tx_search_result = None;
                            }
                            KeyCode::Char('r') => {
                                app.refresh().await?;
                            }
                            KeyCode::Char('a') => {
                                app.auto_refresh_enabled = !app.auto_refresh_enabled;
                            }
                            KeyCode::Char('n') => {
                                if app.has_more_pages {
                                    app.load_next_page().await?;
                                }
                            }
                            _ => {}
                        },
                        View::BlockDetails(_) => match key.code {
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Esc => {
                                app.view = View::BlocksList;
                                app.block_focus = BlockDetailsFocus::Block;
                            }
                            KeyCode::Tab => {
                                if matches!(app.block_focus, BlockDetailsFocus::Block)
                                    && app.selected_tx_id().is_some()
                                {
                                    app.block_focus = BlockDetailsFocus::Transactions;
                                } else {
                                    app.block_focus = BlockDetailsFocus::Block;
                                }
                            }
                            KeyCode::Char('c') => {
                                if let Some(tx_id) = app.selected_tx_id() {
                                    app.copy_tx_id(&tx_id);
                                }
                            }
                            KeyCode::Char('?') => app.open_help(),
                            KeyCode::Enter => {
                                if matches!(app.block_focus, BlockDetailsFocus::Transactions) {
                                    if let (Some(block_idx), Some(tx_idx)) =
                                        (app.list_state.selected(), app.selected_tx_index())
                                    {
                                        app.open_transaction_detail(block_idx, tx_idx).await?;
                                    }
                                }
                            }
                            KeyCode::Down => {
                                if app.block_focus == BlockDetailsFocus::Transactions {
                                    app.move_tx_selection(1);
                                } else if let Some(idx) = app.move_selection_down() {
                                    app.view = View::BlockDetails(idx);
                                    app.sync_tx_list_selection();
                                    if app.should_auto_fetch_more() {
                                        app.load_next_page().await?;
                                    }
                                }
                            }
                            KeyCode::Up => {
                                if app.block_focus == BlockDetailsFocus::Transactions {
                                    app.move_tx_selection(-1);
                                } else if let Some(idx) = app.move_selection_up() {
                                    app.view = View::BlockDetails(idx);
                                    app.sync_tx_list_selection();
                                }
                            }
                            KeyCode::PageDown => {
                                if let Some(idx) = app.page_down() {
                                    app.view = View::BlockDetails(idx);
                                    app.sync_tx_list_selection();
                                    if app.should_auto_fetch_more() {
                                        app.load_next_page().await?;
                                    }
                                }
                            }
                            KeyCode::PageUp => {
                                if let Some(idx) = app.page_up() {
                                    app.view = View::BlockDetails(idx);
                                    app.sync_tx_list_selection();
                                }
                            }
                            _ => {}
                        },
                        View::TransactionDetails { block_idx, .. } => match key.code {
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Esc => {
                                app.view = View::BlockDetails(*block_idx);
                                app.block_focus = BlockDetailsFocus::Transactions;
                                app.sync_tx_list_selection();
                            }
                            KeyCode::Char('c') => {
                                if let Some(detail) = app.tx_detail.clone() {
                                    app.copy_tx_id(&detail.tx_id);
                                }
                            }
                            KeyCode::Char('?') => app.open_help(),
                            _ => {}
                        },
                        View::TransactionSearch => match key.code {
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Esc => app.view = View::BlocksList,
                            KeyCode::Char('?') => app.open_help(),
                            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                app.tx_search_input.clear();
                                app.tx_search_result = None;
                            }
                            KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                if let Some(text) = app.read_clipboard_text() {
                                    app.tx_search_input.push_str(&text);
                                }
                            }
                            KeyCode::Enter => {
                                if !app.tx_search_input.is_empty() {
                                    let search_input = app.tx_search_input.clone();
                                    app.search_transaction(&search_input).await?;
                                }
                            }
                            KeyCode::Backspace => {
                                app.tx_search_input.pop();
                            }
                            KeyCode::Char(c) => {
                                app.tx_search_input.push(c);
                            }
                            _ => {}
                        },
                        View::Help => match key.code {
                            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('?') => {
                                app.close_help()
                            }
                            _ => {}
                        },
                    }
                }
                Event::Paste(data) => {
                    app.clear_status_if_needed();
                    app.note_user_action();
                    if matches!(app.view, View::TransactionSearch) {
                        app.tx_search_input.push_str(&data);
                    }
                }
                _ => {}
            }
        }

        if last_tick.elapsed() >= tick_rate {
            // Auto-reconnect if disconnected
            if app.should_retry_connection() {
                let _ = app.attempt_reconnect().await; // Don't fail on reconnect error
            }

            // Auto-refresh blocks if connected
            if app.should_auto_refresh_blocks(auto_refresh_interval) {
                let _ = app.refresh().await;
            }
            last_tick = Instant::now();
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI args
    let args = Args::parse();

    // Establish connection before touching the terminal so connection failures print normally.
    let app = App::new(args.server, args.fail_fast).await?;

    // Setup terminal with drop guard so panics/errors restore the TTY.
    let mut terminal = TerminalGuard::new()?;

    // Run app
    run_app(terminal.terminal(), app).await
}
fn render_help_menu(f: &mut Frame, area: Rect) {
    let sections = vec![
        (
            "Connection Status",
            vec![
                "● Connected       Server is reachable",
                "● Disconnected    Lost connection (will auto-retry every 5s)",
                "● Never Connected Still trying initial connection",
                "● Reconnecting    Attempting to reconnect now",
            ],
        ),
        (
            "Blocks List",
            vec![
                "↑          Move selection up",
                "↓          Move selection down (auto-loads older blocks)",
                "PgUp       Jump up by 20 blocks",
                "PgDn       Jump down by 20 blocks (auto-loads older)",
                "Enter      View selected block", "c          Copy selected block id",
                "t          Open transaction search", "r          Refresh newest page",
                "a          Toggle auto-refresh", "n          Fetch next page now",
                "?          Show this help", "q          Quit",
            ],
        ),
        (
            "Block Details",
            vec![
                "ESC        Back to list", "PgUp/PgDn  Jump to prev/next block",
                "Tab        Focus/unfocus transaction list", "↑/↓ (tx)   Move tx selection",
                "Enter      View selected transaction",
                "c          Copy highlighted tx id (tx focus)", "?          Show this help",
                "q          Quit",
            ],
        ),
        (
            "Transaction Search",
            vec![
                "ESC        Back to list", "Enter      Search for TX (prefix ok)",
                "Ctrl+V/Ctrl+Shift+V  Paste clipboard", "Ctrl+C     Clear input",
                "?          Show this help", "q          Quit",
            ],
        ),
        (
            "Transaction Details",
            vec![
                "ESC        Back to block", "c          Copy transaction id",
                "?          Show this help", "q          Quit",
            ],
        ),
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        "Keyboard Shortcuts",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    for (title, entries) in sections {
        lines.push(Line::from(Span::styled(
            title,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        for entry in entries {
            lines.push(Line::from(Span::raw(format!("  {}", entry))));
        }
        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        "Blocks you've already fetched stay cached locally; scrolling never re-downloads them.",
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(Span::styled(
        "Press ESC, q, or ? to close this help.",
        Style::default().fg(Color::DarkGray),
    )));

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Help")
                .title_alignment(Alignment::Center),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, area);
}
