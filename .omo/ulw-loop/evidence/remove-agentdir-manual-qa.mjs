import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AutoRAGAgent } from '../../../src/agent/agent.ts';

const root = mkdtempSync(join(tmpdir(), 'autorag-real-dir-qa-'));
try {
  const docs = join(root, 'docs');
  mkdirSync(docs, { recursive: true });
  writeFileSync(join(docs, 'refunds.txt'), 'Refund approvals require manager review.\nEscalate refunds over 500.\n');
  const agent = new AutoRAGAgent({ searchPaths: [docs], workspacePath: root, memoryPath: join(root, 'memory.json') });
  const response = await agent.searchDocuments('Refund approvals', { topK: 1 });
  const pass = response.results.length === 1 && response.answer.includes('Refund approvals require manager review') && !JSON.stringify(response).includes(root);
  console.log(JSON.stringify({ pass, searched: response.searched, answer: response.answer, leakedRoot: JSON.stringify(response).includes(root) }, null, 2));
  if (!pass) process.exit(1);
} finally {
  rmSync(root, { recursive: true, force: true });
  console.log(`cleanup: removed ${root}`);
}
