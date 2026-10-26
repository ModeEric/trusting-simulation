"""Read pinned HTML table text without executing markup or JavaScript."""
from html.parser import HTMLParser


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.cell = None
        self.tables = {}

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            if self.depth:
                raise ValueError('Nested tables are unsupported')
            self.depth = 1
            self.name = dict(attrs).get('id')
            self.rows = []
        elif self.depth and tag == 'tr':
            self.row = []
        elif self.depth and tag in ('td', 'th'):
            self.cell = []

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if self.depth and tag in ('td', 'th'):
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif self.depth and tag == 'tr':
            self.rows.append(self.row)
        elif tag == 'table':
            if not self.depth or self.name in self.tables:
                raise ValueError('Malformed or duplicate table')
            self.tables[self.name] = self.rows
            self.depth = 0


def read_tables(path):
    reader = TableReader()
    reader.feed(path.read_text())
    reader.close()
    return reader.tables
