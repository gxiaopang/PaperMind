import { test, expect } from '@playwright/test'
import { fillEmailCode } from './emailCode'

test('M5 全链路工作台：注册验证、双栏解析预览（公式/表格/安全净化）、流式问答与重生成草稿替换、引用跳页高亮与缺失 bbox 退化、配额错误、会话过期与管理台', async ({
  page,
}) => {
  const email = `m5-user-${Date.now()}@example.org`
  const password = 'Browser test password 123!'

  // 1. 获取邮箱验证码 -> 注册已验证的新账户 -> 登录
  await page.goto('/')
  await page.getByRole('button', { name: '免费注册' }).click()
  await page.getByLabel('邮箱地址').fill(email)
  await fillEmailCode(page, email)
  await page.getByLabel('密码', { exact: true }).fill(password)
  await page.getByRole('button', { name: '创建账户', exact: true }).click()
  await expect(page.getByRole('status')).toContainText('注册成功，邮箱已验证')

  await page.getByLabel('邮箱地址').fill(email)
  await page.getByLabel('密码', { exact: true }).fill(password)
  await page.getByRole('button', { name: '登录', exact: true }).click()
  await expect(page.getByRole('heading', { name: '我的知识库' })).toBeVisible()

  // 2. 创建知识库并上传 PDF，通过 E2E 后台 worker 完成解析与入库
  await page.getByRole('button', { name: '新建知识库' }).click()
  await page.getByLabel('名称', { exact: true }).fill('M5 深度阅读库')
  await page.getByRole('button', { name: '创建', exact: true }).click()
  await page.getByRole('button', { name: 'M5 深度阅读库', exact: true }).click()

  await page.getByLabel('PDF 文件', { exact: true }).setInputFiles(process.env.PAPERMIND_E2E_WORKBENCH_PDF!)
  await page.getByRole('checkbox').check()
  await page.getByRole('button', { name: '上传并入库' }).click()

  // 等待后台入库完成变为“已入库”
  await expect(page.getByText('已入库', { exact: true })).toBeVisible({ timeout: 20000 })
  await expect(page.getByRole('heading', { name: 'Attention Is All You Need (M5 Benchmark)' })).toBeVisible()

  // 3. 打开论文详情双栏预览：检查 Markdown 表格、KaTeX 公式、安全净化（无 script 注入）与右侧 PDF 预览
  await page.getByRole('button', { name: /双栏阅读与原文预览/ }).click()
  await expect(page.getByTestId('paper-detail-page')).toBeVisible()
  await expect(page.locator('.katex').first()).toBeVisible()
  await expect(page.locator('table').first()).toContainText('BLEU')
  // 确保危险脚本和危险协议被净化拦截
  expect(await page.locator('.markdown-body script').count()).toBe(0)
  expect(await page.evaluate(() => (window as unknown as { __xss_executed?: boolean }).__xss_executed)).toBeUndefined()
  await expect(page.locator('.unsafe-link-disabled')).toBeVisible()
  await expect(page.getByTestId('pdf-preview')).toBeVisible()
  await expect(page.getByTestId('pdf-page-indicator')).toContainText('第 1 / 2 页')

  // 切换到结构化分块并点击定位
  await page.getByRole('tab', { name: /结构化分块与 PDF 定位/ }).click()
  await page.getByRole('button', { name: /定位第 1 页（含高亮框）/ }).first().click()
  await expect(page.getByTestId('pdf-bbox-box').first()).toBeVisible()
  await page.screenshot({ path: 'test-results/m5-paper-detail.png', fullPage: true })

  // 4. 从论文详情一键进入问答工作台
  await page.getByRole('button', { name: '针对此论文提问' }).click()
  await expect(page.getByTestId('chat-workbench')).toBeVisible()

  // 发起包含重生成草稿替换 + 双引用（[1] 含 bbox，[2] 无 bbox 位于第 2 页）的提问
  await page.getByLabel('提问内容').fill('请触发重生成草稿并对比缩放点积注意力与实验结果？')
  await page.getByRole('button', { name: '发送提问' }).click()

  // 等待流式回答完成并持久化展示
  await expect(page.getByTestId('final-answer-text').first()).toContainText('缩放点积注意力通过点积除以根号维度计算权重', {
    timeout: 20000,
  })
  // 确保第一版未通过校验的草稿文本不会残留在最终回答中
  await expect(page.getByTestId('final-answer-text').first()).not.toContainText('这是一版未带引用的草稿')

  // 检查右侧检索与校验进度面板（在打开 PDF 预览抽屉之前验证）
  await expect(page.getByTestId('progress-panel')).toBeVisible()
  await expect(page.getByTestId('progress-retrieval-count')).toContainText('1 / 3')
  await expect(page.getByTestId('progress-generation-count')).toContainText('2 / 2')
  await expect(page.getByTestId('progress-verify')).toContainText('LLM 核验通过')

  // 5. 引用徽章悬停预览、点击 [1] 跳页并渲染 bbox 高亮框
  const badge1 = page.getByTestId('citation-badge-1').first()
  await badge1.hover()
  await expect(page.getByTestId('citation-popover-1')).toContainText('支持坐标框高亮')
  await badge1.click()

  const drawer = page.getByTestId('chat-pdf-drawer')
  await expect(drawer).toBeVisible()
  await expect(drawer.getByTestId('pdf-page-indicator')).toContainText('第 1 / 2 页')
  await expect(drawer.getByTestId('pdf-highlight-status')).toContainText('已高亮 1 处区域')
  await expect(drawer.getByTestId('pdf-bbox-box')).toHaveCount(1)
  await page.screenshot({ path: 'test-results/m5-chat-citation-bbox.png', fullPage: true })

  // 6. 点击缺失 bbox 的引用 [2]，验证仅跳到第 2 页且不渲染错误高亮框
  const badge2 = page.getByTestId('citation-badge-2').first()
  await badge2.click()
  await expect(drawer.getByTestId('pdf-page-indicator')).toContainText('第 2 / 2 页')
  await expect(drawer.getByTestId('pdf-highlight-status')).toContainText('页级定位（无坐标框）')
  await expect(drawer.getByTestId('pdf-bbox-box')).toHaveCount(0)

  // 关闭 PDF 预览抽屉后，检索与校验进度面板恢复可见
  await page.getByRole('button', { name: '关闭 PDF 预览' }).click()
  await expect(page.getByTestId('progress-panel')).toBeVisible()

  // 点赞反馈
  await page.getByRole('button', { name: '回答有帮助' }).first().click()
  await expect(page.getByRole('button', { name: '回答有帮助' }).first()).toHaveClass(/active/)

  // 7. 个人设置与用量页面
  await page.getByRole('button', { name: '设置与用量' }).click()
  await expect(page.getByTestId('settings-usage-page')).toBeVisible()
  await page.getByLabel('回答详细程度').selectOption('detailed')
  await page.getByRole('button', { name: '保存个人设置' }).click()
  await expect(page.getByRole('status')).toContainText('个人回答设置已保存')
  await expect(page.getByTestId('quota-meters')).toContainText('今日模型 Token 用量')

  // 8. 退出普通用户，登录管理员验证管理台与配额不足提示
  await page.getByRole('button', { name: '退出登录' }).click()
  await page.getByLabel('邮箱地址').fill('m5-admin@example.org')
  await page.getByLabel('密码', { exact: true }).fill(password)
  await page.getByRole('button', { name: '登录', exact: true }).click()

  await page.getByRole('button', { name: '系统管理' }).click()
  await expect(page.getByTestId('admin-console-page')).toBeVisible()
  await expect(page.getByTestId('admin-queue-status')).toBeVisible()
  await expect(page.getByTestId('admin-eval-section')).toContainText('hybrid_rerank')
  await expect(page.getByRole('heading', { name: '问答真实冒烟（m3-gold-v7）' })).toBeVisible()
  await expect(page.getByTestId('admin-eval-section')).toContainText('各场景分批通过')
  await expect(page.getByTestId('admin-eval-section')).toContainText('回答内容支持率与连续稳定性仍待验收')

  // 将刚才注册用户的每日模型 Token 上限调整为 1，验证配额不足错误展示
  const userRow = page.getByTestId(`admin-user-row-${email}`)
  await expect(userRow).toBeVisible()
  await userRow.getByRole('button', { name: '调整配额' }).click()
  await page.getByLabel('每日模型 Token 上限').fill('1')
  await page.getByRole('button', { name: '保存配额' }).click()
  await page.screenshot({ path: 'test-results/m5-admin-console.png', fullPage: true })

  // 切回普通用户验证配额不足提示与会话过期提示
  await page.getByRole('button', { name: '退出登录' }).click()
  await page.getByLabel('邮箱地址').fill(email)
  await page.getByLabel('密码', { exact: true }).fill(password)
  await page.getByRole('button', { name: '登录', exact: true }).click()

  await page.getByRole('button', { name: '问答工作台' }).click()
  await page.getByLabel('提问内容').fill('配额耗尽后还能提问吗？')
  await page.getByRole('button', { name: '发送提问' }).click()
  await expect(page.getByTestId('chat-error-banner')).toContainText('今日模型 token 额度不足')

  // 模拟会话 Cookie 失效后触发会话过期引导重新登录
  await page.context().clearCookies()
  await page.getByRole('button', { name: '设置与用量' }).click()
  await expect(page.getByTestId('session-expired-banner')).toContainText('会话已过期，请重新登录')
})
