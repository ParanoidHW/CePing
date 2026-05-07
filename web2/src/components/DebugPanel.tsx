/**
 * Debug panel component for collecting debug information.
 * 
 * Features:
 * - Display current state (workload, model, hardware, strategy)
 * - Display recent API requests/responses
 * - Display console errors
 * - Export debug info as JSON
 * - Copy to clipboard
 * 
 * Trigger: Ctrl+Shift+D or click "🔧Debug" button
 */

import { useState, useEffect } from 'react'
import { Modal, Tabs, Button, Space, message, Typography, Card, Descriptions, List, Tag } from 'antd'
import { CopyOutlined, DownloadOutlined, BugOutlined } from '@ant-design/icons'
import { generateDebugInfo, copyToClipboard, downloadAsFile } from '@/utils/debugExport'
import { errorCapture } from '@/utils/errorCapture'
import { requestCapture } from '@/utils/requestCapture'
import type { DebugInfo } from '@/utils/debugExport'
import type { CapturedError } from '@/utils/errorCapture'
import type { CapturedRequest } from '@/utils/requestCapture'
import type { HardwareSchema, StrategySchema } from '@/types'

const { Text, Paragraph } = Typography

interface DebugPanelProps {
  visible: boolean
  onClose: () => void
  workload: string | null
  model: string | null
  hardware: HardwareSchema
  strategy: StrategySchema
  params: Record<string, unknown>
}

export function DebugPanel({
  visible,
  onClose,
  workload,
  model,
  hardware,
  strategy,
  params,
}: DebugPanelProps) {
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null)
  const [errors, setErrors] = useState<CapturedError[]>([])
  const [requests, setRequests] = useState<CapturedRequest[]>([])

  useEffect(() => {
    if (visible) {
      const info = generateDebugInfo(workload, model, hardware, strategy, params)
      setDebugInfo(info)
      setErrors(errorCapture.getErrors())
      setRequests(requestCapture.getRequests())
    }
  }, [visible, workload, model, hardware, strategy, params])

  const handleCopy = async () => {
    if (!debugInfo) return
    const json = JSON.stringify(debugInfo, null, 2)
    const success = await copyToClipboard(json)
    if (success) {
      message.success('Debug info copied to clipboard')
    } else {
      message.error('Failed to copy to clipboard')
    }
  }

  const handleDownload = () => {
    if (!debugInfo) return
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    downloadAsFile(debugInfo, `debug-info-${timestamp}.json`)
    message.success('Debug info downloaded')
  }

  const StateTab = () => (
    <Card size="small">
      <Descriptions column={1} size="small">
        <Descriptions.Item label="Workload">{workload || 'Not selected'}</Descriptions.Item>
        <Descriptions.Item label="Model">{model || 'Not selected'}</Descriptions.Item>
        <Descriptions.Item label="Hardware">
          <Paragraph style={{ marginBottom: 0 }} copyable>
            {JSON.stringify(hardware, null, 2)}
          </Paragraph>
        </Descriptions.Item>
        <Descriptions.Item label="Strategy">
          <Paragraph style={{ marginBottom: 0 }} copyable>
            {JSON.stringify(strategy, null, 2)}
          </Paragraph>
        </Descriptions.Item>
        <Descriptions.Item label="Params">
          <Paragraph style={{ marginBottom: 0 }} copyable>
            {JSON.stringify(params, null, 2)}
          </Paragraph>
        </Descriptions.Item>
      </Descriptions>
    </Card>
  )

  const RequestsTab = () => (
    <List
      size="small"
      dataSource={requests}
      renderItem={(req) => (
        <List.Item>
          <Card size="small" style={{ width: '100%' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Space>
                <Tag color={req.status && req.status < 400 ? 'success' : 'error'}>
                  {req.method}
                </Tag>
                <Text code>{req.url}</Text>
                {req.status && <Tag>{req.status}</Tag>}
              </Space>
              {req.request_body !== undefined && req.request_body !== null && (
                <Paragraph
                  style={{ marginBottom: 0 }}
                  copyable
                  ellipsis={{ rows: 3 }}
                >
                  <Text type="secondary">Request:</Text> {JSON.stringify(req.request_body, null, 2)}
                </Paragraph>
              )}
              {req.response_body !== undefined && req.response_body !== null && (
                <Paragraph
                  style={{ marginBottom: 0 }}
                  copyable
                  ellipsis={{ rows: 3 }}
                >
                  <Text type="secondary">Response:</Text> {JSON.stringify(req.response_body, null, 2)}
                </Paragraph>
              )}
              {req.error && (
                <Paragraph style={{ marginBottom: 0 }} type="danger">
                  Error: {req.error}
                </Paragraph>
              )}
            </Space>
          </Card>
        </List.Item>
      )}
    />
  )

  const ErrorsTab = () => (
    <List
      size="small"
      dataSource={errors}
      renderItem={(err) => (
        <List.Item>
          <Card size="small" style={{ width: '100%' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Space>
                <Tag color="error">{err.type}</Tag>
                <Text type="secondary">{err.timestamp}</Text>
              </Space>
              <Paragraph style={{ marginBottom: 0 }} type="danger">
                {err.message}
              </Paragraph>
              {err.stack && (
                <Paragraph
                  style={{ marginBottom: 0 }}
                  copyable
                  ellipsis={{ rows: 3 }}
                >
                  <Text type="secondary">Stack:</Text> {err.stack}
                </Paragraph>
              )}
              {err.filename && (
                <Text type="secondary">
                  {err.filename}:{err.lineno}:{err.colno}
                </Text>
              )}
            </Space>
          </Card>
        </List.Item>
      )}
    />
  )

  const ExportTab = () => (
    <Space direction="vertical" style={{ width: '100%' }}>
      <Card size="small">
        <Paragraph>
          Click the buttons below to copy or download debug information.
        </Paragraph>
        <Space>
          <Button icon={<CopyOutlined />} onClick={handleCopy}>
            Copy to Clipboard
          </Button>
          <Button icon={<DownloadOutlined />} onClick={handleDownload}>
            Download JSON
          </Button>
        </Space>
      </Card>
      <Card size="small" title="Debug Info Preview">
        <Paragraph
          copyable
          ellipsis={{ rows: 10, expandable: true }}
          style={{ marginBottom: 0, fontFamily: 'monospace' }}
        >
          {debugInfo ? JSON.stringify(debugInfo, null, 2) : 'No debug info'}
        </Paragraph>
      </Card>
    </Space>
  )

  return (
    <Modal
      title={
        <Space>
          <BugOutlined />
          Debug Information
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={900}
      footer={[
        <Button key="copy" icon={<CopyOutlined />} onClick={handleCopy}>
          Copy
        </Button>,
        <Button key="download" icon={<DownloadOutlined />} onClick={handleDownload}>
          Export JSON
        </Button>,
        <Button key="close" onClick={onClose}>
          Close
        </Button>,
      ]}
    >
      <Tabs
        items={[
          {
            key: 'state',
            label: 'State',
            children: <StateTab />,
          },
          {
            key: 'requests',
            label: `Requests (${requests.length})`,
            children: <RequestsTab />,
          },
          {
            key: 'errors',
            label: `Errors (${errors.length})`,
            children: <ErrorsTab />,
          },
          {
            key: 'export',
            label: 'Export',
            children: <ExportTab />,
          },
        ]}
      />
    </Modal>
  )
}