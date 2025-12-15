import json
import gradio as gr
import difflib
from typing import Tuple

class ErrorCaseVisualizer:
    def __init__(self, error_cases_path: str):
        """Initialize the visualizer with error cases data"""
        self.error_cases = []
        self.load_error_cases(error_cases_path)

    def load_error_cases(self, path: str):
        """Load error cases from jsonl file"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.error_cases.append(json.loads(line))
        print(f"Loaded {len(self.error_cases)} error cases")

    def get_char_diff_html(self, original: str, generated: str) -> Tuple[str, str]:
        """
        Generate HTML with character-level differences highlighted.
        Returns (original_html, generated_html)
        """
        # Use SequenceMatcher for character-level comparison
        matcher = difflib.SequenceMatcher(None, original, generated)

        original_html = []
        generated_html = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            original_text = original[i1:i2]
            generated_text = generated[j1:j2]

            if tag == 'equal':
                # Text is the same
                original_html.append(f'<span style="color: black;">{self._escape_html(original_text)}</span>')
                generated_html.append(f'<span style="color: black;">{self._escape_html(generated_text)}</span>')
            elif tag == 'replace':
                # Text was replaced
                original_html.append(f'<span style="background-color: #ffcccc; color: #cc0000; font-weight: bold;">{self._escape_html(original_text)}</span>')
                generated_html.append(f'<span style="background-color: #ccffcc; color: #009900; font-weight: bold;">{self._escape_html(generated_text)}</span>')
            elif tag == 'delete':
                # Text was deleted from original
                original_html.append(f'<span style="background-color: #ffcccc; color: #cc0000; font-weight: bold; text-decoration: line-through;">{self._escape_html(original_text)}</span>')
            elif tag == 'insert':
                # Text was inserted in generated
                generated_html.append(f'<span style="background-color: #ccffcc; color: #009900; font-weight: bold;">{self._escape_html(generated_text)}</span>')

        return ''.join(original_html), ''.join(generated_html)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters and preserve whitespace"""
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('\n', '<br>')
        text = text.replace(' ', '&nbsp;')
        return text

    def get_word_diff_html(self, original: str, generated: str) -> Tuple[str, str]:
        """
        Generate HTML with word-level differences highlighted.
        Returns (original_html, generated_html)
        """
        original_words = original.split()
        generated_words = generated.split()

        matcher = difflib.SequenceMatcher(None, original_words, generated_words)

        original_html = []
        generated_html = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            original_text = ' '.join(original_words[i1:i2])
            generated_text = ' '.join(generated_words[j1:j2])

            if tag == 'equal':
                original_html.append(f'<span style="color: black;">{self._escape_html(original_text)}</span>')
                generated_html.append(f'<span style="color: black;">{self._escape_html(generated_text)}</span>')
            elif tag == 'replace':
                original_html.append(f'<span style="background-color: #ffcccc; color: #cc0000; font-weight: bold;">{self._escape_html(original_text)}</span>')
                generated_html.append(f'<span style="background-color: #ccffcc; color: #009900; font-weight: bold;">{self._escape_html(generated_text)}</span>')
            elif tag == 'delete':
                original_html.append(f'<span style="background-color: #ffcccc; color: #cc0000; font-weight: bold; text-decoration: line-through;">{self._escape_html(original_text)}</span>')
            elif tag == 'insert':
                generated_html.append(f'<span style="background-color: #ccffcc; color: #009900; font-weight: bold;">{self._escape_html(generated_text)}</span>')

            # Add space between words
            if i2 < len(original_words):
                original_html.append(' ')
            if j2 < len(generated_words):
                generated_html.append(' ')

        return ''.join(original_html), ''.join(generated_html)

    def display_error_case(self, case_index: int, diff_mode: str = "character") -> Tuple[str, str, str, str]:
        """
        Display a specific error case with highlighted differences.

        Args:
            case_index: Index of the error case to display (0-based)
            diff_mode: "character" for char-level diff, "word" for word-level diff

        Returns:
            (stats_html, original_html, generated_html, info_message)
        """
        if case_index < 0 or case_index >= len(self.error_cases):
            return "Invalid case index", "", "", f"Please select a case between 0 and {len(self.error_cases)-1}"

        case = self.error_cases[case_index]

        # Generate statistics
        stats_html = f"""
        <div style="padding: 15px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px;">
            <h3 style="margin-top: 0;">Error Case #{case_index + 1} / {len(self.error_cases)}</h3>
            <p><strong>Token Accuracy:</strong> {case['token_accuracy']:.4f} ({case['token_accuracy']*100:.2f}%)</p>
            <p><strong>Correct Tokens:</strong> {case['correct_tokens']} / {case['total_tokens']}</p>
            <p><strong>Error Tokens:</strong> {case['total_tokens'] - case['correct_tokens']}</p>
        </div>
        """

        # Generate diff visualization
        original = case['original_text_decoded']
        generated = case['generated_text']

        if diff_mode == "character":
            original_html, generated_html = self.get_char_diff_html(original, generated)
        else:  # word mode
            original_html, generated_html = self.get_word_diff_html(original, generated)

        # Wrap in styled divs
        original_html = f"""
        <div style="padding: 15px; background-color: #fff5f5; border: 2px solid #ffcccc; border-radius: 5px; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; line-height: 1.6; max-width: 100%; overflow-x: auto;">
            <h4 style="margin-top: 0; color: #cc0000;">Original Text (Decoded from Tokens)</h4>
            {original_html}
        </div>
        """

        generated_html = f"""
        <div style="padding: 15px; background-color: #f5fff5; border: 2px solid #ccffcc; border-radius: 5px; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; line-height: 1.6; max-width: 100%; overflow-x: auto;">
            <h4 style="margin-top: 0; color: #009900;">Generated Text (Model Output)</h4>
            {generated_html}
        </div>
        """

        legend_html = """
        <div style="padding: 10px; background-color: #ffffcc; border-radius: 5px; margin-top: 10px;">
            <strong>Legend:</strong><br>
            <span style="background-color: #ffcccc; color: #cc0000; padding: 2px 5px;">Red/Pink</span> = Original text (deleted/changed)<br>
            <span style="background-color: #ccffcc; color: #009900; padding: 2px 5px;">Green</span> = Generated text (inserted/changed)<br>
            <span style="color: black; padding: 2px 5px;">Black</span> = Matching text
        </div>
        """

        info_message = f"Displaying case {case_index + 1} of {len(self.error_cases)} (Diff mode: {diff_mode})"

        return stats_html, original_html, generated_html, legend_html, info_message


def create_gradio_interface(error_cases_path: str):
    """Create and launch the Gradio interface"""

    visualizer = ErrorCaseVisualizer(error_cases_path)

    with gr.Blocks(title="TextVAE Error Case Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # TextVAE Error Case Visualizer

            This tool helps you visualize and compare error cases from the TextVAE model.
            Use the slider to navigate through different error cases and see where the model made mistakes.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                case_slider = gr.Slider(
                    minimum=0,
                    maximum=len(visualizer.error_cases) - 1,
                    step=1,
                    value=0,
                    label=f"Select Error Case (0 to {len(visualizer.error_cases) - 1})"
                )
            with gr.Column(scale=1):
                diff_mode = gr.Radio(
                    choices=["character", "word"],
                    value="character",
                    label="Diff Mode",
                    info="Choose character-level or word-level comparison"
                )

        refresh_btn = gr.Button("Refresh Display", variant="primary")

        info_box = gr.Textbox(label="Status", interactive=False)

        stats_display = gr.HTML(label="Statistics")

        legend_display = gr.HTML(label="Legend")

        with gr.Row():
            with gr.Column():
                original_display = gr.HTML(label="Original Text")
            with gr.Column():
                generated_display = gr.HTML(label="Generated Text")

        # Event handlers
        def update_display(case_idx, mode):
            return visualizer.display_error_case(int(case_idx), mode)

        refresh_btn.click(
            fn=update_display,
            inputs=[case_slider, diff_mode],
            outputs=[stats_display, original_display, generated_display, legend_display, info_box]
        )

        case_slider.change(
            fn=update_display,
            inputs=[case_slider, diff_mode],
            outputs=[stats_display, original_display, generated_display, legend_display, info_box]
        )

        diff_mode.change(
            fn=update_display,
            inputs=[case_slider, diff_mode],
            outputs=[stats_display, original_display, generated_display, legend_display, info_box]
        )

        # Load first case on startup
        demo.load(
            fn=update_display,
            inputs=[case_slider, diff_mode],
            outputs=[stats_display, original_display, generated_display, legend_display, info_box]
        )

    return demo


if __name__ == "__main__":
    ERROR_CASES_PATH = "./reports/error_cases.jsonl"

    print("Starting Error Case Visualizer...")
    demo = create_gradio_interface(ERROR_CASES_PATH)
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
