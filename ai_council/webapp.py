import warnings
warnings.filterwarnings('ignore')

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import threading
import sys
from io import StringIO
from datetime import datetime
import time

from ai_council.council import *
from ai_council.prompts import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from ai_council.vector import *

class AICouncilUI:
    def __init__(self):
        self.vs = None
        self.is_processing = False
        self.logs = []
        
        # Create UI components
        self.create_ui()
        
        # Initialize vector DB in background
        self.initialize_vector_db()
    
    def create_ui(self):
        """Create the Jupyter notebook UI"""
        
        # Status indicator
        self.status_html = widgets.HTML(
            value=self._render_status('initializing', 'Initializing vector database...')
        )
        
        # Chat display area
        self.chat_html = widgets.HTML(
            value=self._render_chat([{
                'type': 'assistant',
                'text': 'Welcome! Initializing AI Council...'
            }])
        )
        
        # Log display area
        self.log_html = widgets.HTML(
            value=self._render_logs([])
        )
        
        # Input area
        self.input_text = widgets.Text(
            placeholder='Type your message here...',
            disabled=True,
            layout=widgets.Layout(width='80%')
        )
        
        self.send_button = widgets.Button(
            description='Send',
            disabled=True,
            button_style='primary',
            layout=widgets.Layout(width='18%')
        )
        
        # Event handlers
        self.send_button.on_click(self.on_send_clicked)
        self.input_text.on_submit(self.on_send_clicked)
        
        # Layout
        input_box = widgets.HBox([self.input_text, self.send_button])
        
        chat_section = widgets.VBox([
            widgets.HTML('<h3 style="margin: 10px 0; color: #667eea;">💬 Chat</h3>'),
            self.chat_html,
            input_box
        ], layout=widgets.Layout(width='65%'))  # Chat takes 65% of width
        
        log_section = widgets.VBox([
            widgets.HTML('<h3 style="margin: 10px 0; color: #667eea;">📋 System Logs</h3>'),
            self.log_html
        ], layout=widgets.Layout(width='33%'))  # Logs take 33% of width
        
        main_layout = widgets.VBox([
            widgets.HTML('''
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white; margin-bottom: 10px;">
                    <h2 style="margin: 0;">🤖 AI Council Chat Interface</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Multi-model AI consensus system</p>
                </div>
            '''),
            self.status_html,
            widgets.HBox([chat_section, log_section], 
                        layout=widgets.Layout(width='100%', justify_content='space-between'))
        ])
        
        display(main_layout)
        
        self.messages = [{
            'type': 'assistant',
            'text': 'Welcome! Initializing AI Council...'
        }]
    
    def _render_status(self, status_type, message):
        """Render status indicator"""
        colors = {
            'ready': '#d4edda',
            'processing': '#fff3cd',
            'initializing': '#fff3cd',
            'error': '#f8d7da'
        }
        icons = {
            'ready': '✅',
            'processing': '⏳',
            'initializing': '⏳',
            'error': '❌'
        }
        
        return f'''
            <div style="padding: 10px; background: {colors.get(status_type, '#fff')}; 
                        border-radius: 5px; margin-bottom: 10px;">
                {icons.get(status_type, '⏳')} <b>Status:</b> {message}
            </div>
        '''
    
    def _render_chat(self, messages):
        """Render chat messages"""
        html = '<div style="height: 400px; overflow-y: auto; border: 2px solid #e0e0e0; border-radius: 10px; padding: 10px; background: white;">'
        
        for msg in messages:
            if msg['type'] == 'user':
                html += f'''
                    <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 12px 16px; border-radius: 18px; 
                                    max-width: 70%; word-wrap: break-word;">
                            {self._escape_html(msg['text'])}
                        </div>
                    </div>
                '''
            else:
                html += f'''
                    <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                        <div style="background: #f5f5f5; color: #333; 
                                    padding: 12px 16px; border-radius: 18px; 
                                    max-width: 70%; word-wrap: break-word;">
                            {self._escape_html(msg['text'])}
                        </div>
                    </div>
                '''
        
        html += '</div>'
        return html
    
    def _render_logs(self, logs):
        """Render log entries with auto-scroll to bottom"""
        # Generate unique ID for this render to ensure scroll happens
        scroll_id = f"log_container_{int(time.time() * 1000)}"
        
        html = f'''
            <div id="{scroll_id}" style="height: 400px; overflow-y: auto; border: 2px solid #e0e0e0; 
                        border-radius: 10px; padding: 10px; background: #1e1e1e; 
                        font-family: 'Courier New', monospace; font-size: 12px;">
        '''
        
        for log in logs:
            color_map = {
                "INFO": "#81c784",
                "WARNING": "#ffb74d",
                "ERROR": "#e57373"
            }
            color = color_map.get(log['level'], "#d4d4d4")
            
            html += f'''
                <div style="margin-bottom: 8px; line-height: 1.5; color: #d4d4d4;">
                    <span style="color: #4fc3f7;">[{log['timestamp']}]</span>
                    <span style="color: {color};">[{log['level']}]</span>
                    {self._escape_html(log['message'])}
                </div>
            '''
        
        html += '</div>'
        
        # Add JavaScript to scroll to bottom
        html += f'''
            <script>
                (function() {{
                    var container = document.getElementById("{scroll_id}");
                    if (container) {{
                        container.scrollTop = container.scrollHeight;
                    }}
                }})();
            </script>
        '''
        
        return html
    
    def _escape_html(self, text):
        """Escape HTML characters and handle **bold** formatting"""
        # First escape HTML characters
        text = str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
        
        # Then convert **text** to <strong>text</strong>
        import re
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        
        return text
    
    def update_status(self, status_type, message):
        """Update status indicator"""
        self.status_html.value = self._render_status(status_type, message)
    
    def update_chat(self):
        """Update chat display"""
        self.chat_html.value = self._render_chat(self.messages)
    
    def update_logs(self):
        """Update log display"""
        self.log_html.value = self._render_logs(self.logs)
    
    def add_log(self, level, message):
        """Add a log entry"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
        self.update_logs()
    
    def initialize_vector_db(self):
        """Initialize vector database in background thread"""
        def init_task():
            try:
                self.add_log("INFO", "Starting AI Council initialization...")
                self.add_log("INFO", "Loading vector database...")
                
                # Capture print statements
                old_stdout = sys.stdout
                sys.stdout = LogCapture(self)
                
                self.vs = get_vector_db()
                
                sys.stdout = old_stdout
                
                self.add_log("INFO", "Vector database loaded successfully")
                self.add_log("INFO", "AI Council ready")
                
                # Update UI
                self.update_status('ready', 'Ready')
                
                self.messages = [{
                    'type': 'assistant',
                    'text': 'AI Council initialized! How can I help you today?'
                }]
                self.update_chat()
                
                # Enable input
                self.input_text.disabled = False
                self.send_button.disabled = False
                
            except Exception as e:
                self.add_log("ERROR", f"Initialization failed: {str(e)}")
                self.update_status('error', f'Error - {str(e)}')
        
        thread = threading.Thread(target=init_task, daemon=True)
        thread.start()
    
    def on_send_clicked(self, b):
        """Handle send button click"""
        user_input = self.input_text.value.strip()
        
        if not user_input or self.is_processing:
            return
        
        # Clear input and disable
        self.input_text.value = ''
        self.is_processing = True
        self.input_text.disabled = True
        self.send_button.disabled = True
        
        # Update status
        self.update_status('processing', 'Processing...')
        
        # Add user message
        self.messages.append({
            'type': 'user',
            'text': user_input
        })
        self.update_chat()
        
        # Process in background thread
        thread = threading.Thread(
            target=self.process_input,
            args=(user_input,),
            daemon=True
        )
        thread.start()
    
    def process_input(self, user_input):
        """Process user input through AI Council workflow"""
        try:
            # Capture print statements
            old_stdout = sys.stdout
            sys.stdout = LogCapture(self)
            
            self.add_log("INFO", f"Processing user input: '{user_input}'")
            
            # Run the workflow
            context = self.vs.similarity_search(user_input)
            self.add_log("INFO", "Context retrieved from vector database")
            
            responses, user_prompt = generate_expert_response(user_input, context)
            
            scoring_matrix = generate_scores(responses, user_prompt)
            
            audit, audit_prompt = generate_audit_report(user_input, responses, scoring_matrix)
            
            scoring_matrix, averages_json, best_response = audited_scoring_matrix(
                audit, scoring_matrix, responses
            )
            
            sys.stdout = old_stdout
            
            self.add_log("INFO", averages_json)
            self.add_log("INFO", best_response)

            
            # Display best response
            self.messages.append({
                'type': 'assistant',
                'text': best_response['text'] if not IS_ONLINE else best_response['text'].content
            })
            self.update_chat()
            
        except Exception as e:
            sys.stdout = old_stdout
            self.add_log("ERROR", f"Processing failed: {str(e)}")
            import traceback
            self.add_log("ERROR", traceback.format_exc())
            
            self.messages.append({
                'type': 'assistant',
                'text': f"Sorry, an error occurred: {str(e)}"
            })
            self.update_chat()
        
        finally:
            # Re-enable input
            self.is_processing = False
            self.input_text.disabled = False
            self.send_button.disabled = False
            
            # Update status
            self.update_status('ready', 'Ready')


class LogCapture:
    """Capture print statements and redirect to log"""
    def __init__(self, ui):
        self.ui = ui
        self.buffer = StringIO()
    
    def write(self, text):
        if text.strip():
            self.ui.add_log("INFO", text.strip())
        self.buffer.write(text)
    
    def flush(self):
        pass
    
    def getvalue(self):
        return self.buffer.getvalue()
