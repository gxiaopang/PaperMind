import fs from 'node:fs'
import { expect, type Page } from '@playwright/test'

export async function fillEmailCode(page: Page, email: string) {
  await page.getByRole('button', { name: '发送验证码', exact: true }).click()
  await expect(page.getByRole('status')).toContainText('验证码邮件已加入发送队列')
  const mailboxPath = process.env.PAPERMIND_E2E_MAILBOX!
  let code = ''
  await expect.poll(() => {
    if (!fs.existsSync(mailboxPath)) return ''
    const items = JSON.parse(fs.readFileSync(mailboxPath, 'utf-8')) as Array<{
      email: string; kind: string; code?: string
    }>
    code = items.reverse().find(m => m.email === email && m.kind === 'email_code')?.code ?? ''
    return code
  }).toMatch(/^[0-9]{6}$/)
  await page.getByLabel('邮箱验证码', { exact: true }).fill(code)
}
